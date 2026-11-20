"""vlm-bridge v2: 模板化三级流水线 + 批量

配置: config.yaml (模型注册表 + 流水线模板)
接口:
  GET  /health
  GET  /models           可用模型
  GET  /templates        可用模板
  POST /analyze          单张 (multipart: file, template, prompt)
  POST /batch            按目录建批 (JSON: template, dir, limit, save_images)
  POST /batch/upload     按上传建批 (multipart: template, files[])
  GET  /batch            任务列表
  GET  /batch/{id}       进度
  GET  /batch/{id}/items 明细(分页)
  GET  /batch/{id}/export.csv
  POST /batch/{id}/cancel
"""
import base64
import csv
import glob
import io
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import urllib.request
import uuid
from datetime import datetime

import torch
import yaml
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("VLM_CONFIG", os.path.join(BASE, "config.yaml"))
JOBS_DIR = os.environ.get("VLM_JOBS_DIR", os.path.join(BASE, "jobs"))
DEVICE = os.environ.get("VLM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "192"))
POOL_LIMIT = {"detect": 2, "segment": 2, "reason": 1}

os.makedirs(JOBS_DIR, exist_ok=True)

CFG = {}
MODELS = {}
TEMPLATES = {}
POOL = {}
POOL_ORDER = []
POOL_LOCK = threading.Lock()
STATE = {"status": "loading", "error": None, "loaded_at": None}


def load_config():
    global CFG, MODELS, TEMPLATES
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}
    MODELS = CFG.get("models", {}) or {}
    TEMPLATES = CFG.get("templates", {}) or {}


# ---------- 图像预处理 (InternVL 官方做法) ----------
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    out = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        out.append(resized_img.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_pixel_values(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    imgs = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(i) for i in imgs])


# ---------- 模型池 ----------
def _evict_for(kind):
    with POOL_LOCK:
        same = [m for m in POOL_ORDER if MODELS.get(m, {}).get("type") == kind]
        while len(same) >= POOL_LIMIT.get(kind, 1):
            victim = same.pop(0)
            POOL.pop(victim, None)
            if victim in POOL_ORDER:
                POOL_ORDER.remove(victim)


def get_model(mid):
    if mid not in MODELS:
        raise KeyError("unknown model: %s" % mid)
    spec = MODELS[mid]
    kind = spec.get("type")
    # 云端模型不加载权重、不占显存，也不进入 LRU 池
    if kind == "reason_api":
        return {"spec": spec}
    with POOL_LOCK:
        if mid in POOL:
            return POOL[mid]
    if kind == "detect":
        from ultralytics import YOLO
        obj = YOLO(spec["path"])
    elif kind == "segment":
        from ultralytics import SAM
        obj = SAM(spec["path"])
    elif kind == "reason":
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=True, use_fast=False)
        dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
        net = AutoModel.from_pretrained(spec["path"], torch_dtype=dtype,
                                        low_cpu_mem_usage=True, trust_remote_code=True).eval().to(DEVICE)
        obj = {"model": net, "tokenizer": tok}
    else:
        raise ValueError("bad model type: %s" % kind)
    _evict_for(kind)
    with POOL_LOCK:
        POOL[mid] = obj
        if mid in POOL_ORDER:
            POOL_ORDER.remove(mid)
        POOL_ORDER.append(mid)
    return obj


def warm(template_id):
    stages = TEMPLATES[template_id]["stages"]
    for name in ("detect", "segment", "reason"):
        if name in stages:
            mid = stages[name]["model"]
            # 云端模型无需预热加载
            if (MODELS.get(mid) or {}).get("type") == "reason_api":
                continue
            get_model(mid)


def call_reason_api(spec, image, context, question):
    """调用云端多模态大模型（OpenAI 兼容 /chat/completions）。

    密钥不写入 config.yaml，由 spec.api_key_env 指定的环境变量提供。
    """
    base_url = (spec.get("base_url") or "").strip().rstrip("/")
    model = (spec.get("model") or "").strip()
    if not base_url or not model:
        raise ValueError("云端模型缺少 base_url 或 model 配置")
    key_env = (spec.get("api_key_env") or "VLM_API_KEY").strip()
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise ValueError("未设置环境变量 %s（云端模型密钥）" % key_env)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": context + "\n" + question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": int(spec.get("max_tokens", MAX_NEW_TOKENS)),
        "temperature": 0,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(spec.get("timeout", 60))) as resp:
        body = json.loads(resp.read().decode("utf-8", "replace"))
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("云端返回结果为空：%s" % json.dumps(body, ensure_ascii=False)[:300])
    return (choices[0].get("message", {}).get("content") or "").strip()


# ---------- 流水线 ----------
def run_pipeline(image, template_id, prompt_override="", save_path=None):
    if template_id not in TEMPLATES:
        raise KeyError("unknown template: %s" % template_id)
    stages = TEMPLATES[template_id]["stages"]
    w, h = image.size
    timings = {}
    detections = []
    vlm_text = ""

    if "detect" in stages:
        spec = stages["detect"]
        yolo = get_model(spec["model"])
        conf = float(spec.get("conf", MODELS.get(spec["model"], {}).get("conf", 0.25)))
        t0 = time.time()
        res = yolo.predict(image, conf=conf, verbose=False)[0]
        names = getattr(res, "names", {}) or {}
        for b in res.boxes:
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            detections.append({"label": names.get(int(b.cls), str(int(b.cls))),
                               "conf": round(float(b.conf), 4),
                               "box": [round(v, 1) for v in xyxy]})
        timings["detect_ms"] = int((time.time() - t0) * 1000)

    if "segment" in stages and detections:
        spec = stages["segment"]
        sam = get_model(spec["model"])
        t1 = time.time()
        try:
            sr = sam(image, bboxes=[d["box"] for d in detections], verbose=False)[0]
            if sr.masks is not None:
                areas = sr.masks.data.reshape(sr.masks.data.shape[0], -1).sum(dim=1).tolist()
                for d, a in zip(detections, areas):
                    d["mask_area_ratio"] = round(a / (w * h), 5)
        except Exception:
            timings["segment_error"] = traceback.format_exc(limit=1)
        timings["segment_ms"] = int((time.time() - t1) * 1000)

    skip_reason = ("detect" in stages) and not detections
    if "reason" in stages and skip_reason:
        vlm_text = "未检出目标，未调用 VLM"
        timings["reason_ms"] = 0
    elif "reason" in stages:
        spec = stages["reason"]
        mid = spec["model"]
        is_api = (MODELS.get(mid) or {}).get("type") == "reason_api"
        question = prompt_override.strip() or spec.get("prompt", "").strip()
        ctx = "检测结果: " + (json.dumps(detections, ensure_ascii=False) if detections else "未检出目标")
        t2 = time.time()
        try:
            if is_api:
                vlm_text = call_reason_api(MODELS[mid], image, ctx, question)
            else:
                net = get_model(mid)
                pv = load_pixel_values(image).to(torch.bfloat16).to(DEVICE)
                reply = net["model"].chat(net["tokenizer"], pv, ctx + "\n" + question,
                                          dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False))
                vlm_text = reply[0] if isinstance(reply, tuple) else reply
        except Exception as exc:
            timings["reason_error"] = traceback.format_exc(limit=2)
            if is_api:
                vlm_text = "云端解释调用失败：%s" % exc
        timings["reason_ms"] = int((time.time() - t2) * 1000)

    if save_path and detections:
        try:
            draw = ImageDraw.Draw(image)
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
            image.save(save_path, quality=88)
        except Exception:
            pass

    return {
        "template": template_id,
        "image_size": [w, h],
        "detections": detections,
        "summary": {
            "count": len(detections),
            "max_conf": max([d["conf"] for d in detections], default=0.0),
            "mask_area_ratio_sum": round(sum(d.get("mask_area_ratio", 0) for d in detections), 5),
        },
        "reasoning": {"text": vlm_text, "model": (stages.get("reason") or {}).get("model", "")},
        "timings": timings,
    }


# ---------- 批量 ----------
BATCH_Q = queue.Queue()
JOB_LOCK = threading.Lock()
CANCEL = set()


def job_dir(jid):
    return os.path.join(JOBS_DIR, jid)


def read_meta(jid):
    p = os.path.join(job_dir(jid), "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_meta(jid, meta):
    p = os.path.join(job_dir(jid), "meta.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def create_job(template, images, save_images=False, title=""):
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    os.makedirs(os.path.join(job_dir(jid), "annotated"), exist_ok=True)
    meta = {
        "id": jid,
        "template": template,
        "title": title or template,
        "save_images": bool(save_images),
        "total": len(images),
        "done": 0,
        "failed": 0,
        "detected": 0,
        "status": "queued",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": None,
        "finished_at": None,
        "elapsed_sec": 0,
        "images": images,
        "error": None,
    }
    write_meta(jid, meta)
    BATCH_Q.put(jid)
    return meta


def run_job(jid):
    meta = read_meta(jid)
    if not meta:
        return
    meta["status"] = "running"
    meta["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_meta(jid, meta)
    items_path = os.path.join(job_dir(jid), "items.jsonl")
    t_start = time.time()
    with open(items_path, "a", encoding="utf-8") as items:
        for idx, path in enumerate(meta["images"]):
            if jid in CANCEL:
                meta["status"] = "cancelled"
                break
            rec = {"index": idx, "path": path, "name": os.path.basename(path)}
            try:
                with Image.open(path) as im:
                    image = im.convert("RGB")
                save_path = None
                if meta["save_images"]:
                    save_path = os.path.join(job_dir(jid), "annotated", os.path.splitext(rec["name"])[0] + ".jpg")
                out = run_pipeline(image, meta["template"], save_path=save_path)
                rec.update({"ok": True, "summary": out["summary"], "reasoning": out["reasoning"],
                            "timings": out["timings"], "detections": out["detections"]})
                if out["summary"]["count"] > 0:
                    meta["detected"] += 1
            except Exception:
                rec.update({"ok": False, "error": traceback.format_exc(limit=2)})
                meta["failed"] += 1
            meta["done"] = idx + 1
            meta["elapsed_sec"] = round(time.time() - t_start, 1)
            items.write(json.dumps(rec, ensure_ascii=False) + "\n")
            items.flush()
            write_meta(jid, meta)
    if meta["status"] != "cancelled":
        meta["status"] = "done"
    meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_sec"] = round(time.time() - t_start, 1)
    write_meta(jid, meta)
    CANCEL.discard(jid)


def worker_loop():
    while True:
        jid = BATCH_Q.get()
        try:
            run_job(jid)
        except Exception:
            meta = read_meta(jid) or {"id": jid}
            meta["status"] = "error"
            meta["error"] = traceback.format_exc(limit=2)
            write_meta(jid, meta)


app = FastAPI(title="vlm-bridge", version="2.0")


@app.on_event("startup")
def _startup():
    def boot():
        try:
            load_config()
            if TEMPLATES:
                first = next(iter(TEMPLATES))
                warm(first)
            STATE["status"] = "ok"
            STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            STATE["status"] = "error"
            STATE["error"] = traceback.format_exc()
    threading.Thread(target=boot, daemon=True).start()
    threading.Thread(target=worker_loop, daemon=True).start()


@app.get("/health")
def health():
    return {"status": STATE["status"], "error": STATE["error"], "device": DEVICE,
            "loaded_at": STATE["loaded_at"], "pool": list(POOL.keys()),
            "templates": list(TEMPLATES.keys()), "models": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {"models": [{"id": k, **v} for k, v in MODELS.items()]}


@app.get("/templates")
def list_templates():
    out = []
    for tid, t in TEMPLATES.items():
        out.append({"id": tid, "name": t.get("name", tid), "stages": list((t.get("stages") or {}).keys())})
    return {"templates": out}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), template: str = Form(""), prompt: str = Form("")):
    if STATE["status"] != "ok":
        return JSONResponse({"detail": "not ready", "error": STATE["error"]}, status_code=503)
    tid = template or next(iter(TEMPLATES))
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        out = run_pipeline(image, tid, prompt_override=prompt)
    except KeyError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    out["ok"] = True
    return out


@app.post("/batch")
async def batch_create(payload: dict = Body(...)):
    tid = payload.get("template") or next(iter(TEMPLATES))
    d = payload.get("dir")
    if not d or not os.path.isdir(d):
        return JSONResponse({"detail": "目录不存在: %s" % d}, status_code=400)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if payload.get("recursive"):
        files = sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in exts and p.is_file())
    else:
        files = sorted(f for f in glob.glob(os.path.join(d, "*")) if os.path.splitext(f)[1].lower() in exts)
    skipped = 0
    if payload.get("skip_processed"):
        seen = _processed_paths()
        before = len(files)
        files = [f for f in files if f not in seen]
        skipped = before - len(files)
    limit = int(payload.get("limit") or 0)
    if limit > 0:
        files = files[:limit]
    if not files:
        return JSONResponse({"detail": "目录下没有待处理图片: %s（已跳过 %d 张历史图片）" % (d, skipped)}, status_code=400)
    meta = create_job(tid, files, payload.get("save_images", False), payload.get("title", ""))
    meta.pop("images", None)
    meta["skipped_processed"] = skipped
    return meta


def _processed_paths() -> set:
    seen = set()
    for name in os.listdir(JOBS_DIR):
        meta = read_meta(name)
        if not meta:
            continue
        for path in meta.get("images") or []:
            seen.add(path)
    return seen


@app.post("/batch/upload")
async def batch_upload(template: str = Form(""), files: list[UploadFile] = File(...), save_images: bool = Form(False)):
    tid = template or next(iter(TEMPLATES))
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    inbox = os.path.join(job_dir(jid), "input")
    os.makedirs(inbox, exist_ok=True)
    saved = []
    for f in files:
        name = os.path.basename(f.filename or "img")
        p = os.path.join(inbox, name)
        with open(p, "wb") as out:
            out.write(await f.read())
        saved.append(p)
    meta = create_job(tid, saved, save_images, "上传批次")
    meta.pop("images", None)
    return meta


@app.get("/batch")
def batch_list():
    out = []
    for d in sorted(os.listdir(JOBS_DIR), reverse=True):
        m = read_meta(d)
        if m:
            out.append({k: m.get(k) for k in ("id", "title", "template", "total", "done", "failed",
                                             "detected", "status", "created_at", "elapsed_sec")})
    return {"jobs": out[:50]}


@app.get("/batch/{jid}")
def batch_status(jid: str):
    m = read_meta(jid)
    if not m:
        return JSONResponse({"detail": "not found"}, status_code=404)
    m.pop("images", None)
    if m["status"] == "running" and m["done"]:
        avg = m["elapsed_sec"] / max(m["done"], 1)
        m["eta_sec"] = round(avg * (m["total"] - m["done"]), 1)
    return m


@app.get("/batch/{jid}/items")
def batch_items(jid: str, offset: int = 0, limit: int = 200):
    p = os.path.join(job_dir(jid), "items.jsonl")
    if not os.path.exists(p):
        return {"items": [], "total": 0}
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    chunk = rows[offset:offset + limit]
    for r in chunk:
        r.pop("detections", None)
    return {"items": chunk, "total": len(rows)}


@app.get("/batch/{jid}/export.csv")
def batch_csv(jid: str):
    p = os.path.join(job_dir(jid), "items.jsonl")
    if not os.path.exists(p):
        return JSONResponse({"detail": "not found"}, status_code=404)

    def gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["序号", "文件名", "路径", "是否成功", "检出数", "最高置信度", "面积占比", "结论", "耗时ms", "错误"])
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                s = r.get("summary") or {}
                t = r.get("timings") or {}
                w.writerow([r.get("index", ""), r.get("name", ""), r.get("path", ""),
                            "是" if r.get("ok") else "否", s.get("count", ""), s.get("max_conf", ""),
                            s.get("mask_area_ratio_sum", ""),
                            (r.get("reasoning") or {}).get("text", "") or "",
                            sum(v for k, v in t.items() if k.endswith("_ms") and isinstance(v, int)),
                            (r.get("error") or "").splitlines()[0] if r.get("error") else ""])
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)

    return StreamingResponse(gen(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=%s.csv" % jid})


@app.post("/batch/{jid}/cancel")
def batch_cancel(jid: str):
    if not read_meta(jid):
        return JSONResponse({"detail": "not found"}, status_code=404)
    CANCEL.add(jid)
    return {"ok": True, "cancelling": jid}
