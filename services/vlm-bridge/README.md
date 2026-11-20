vlm-bridge
==========

独立推理服务（YOLO + SAM2 + InternVL2 三级流水线），由 YWA「智能分析」页通过 /api/pipeline/* 代理调用。

运行
----
  pip install fastapi uvicorn pyyaml pillow transformers==4.44.2 accelerate timm einops sentencepiece
  VLM_MODEL_PATH=/path/to/InternVL2-1B uvicorn app:app --host 127.0.0.1 --port 8002

配置
----
config.yaml: 模型注册表 models + 流水线模板 templates（可加模型、改提示词，不改代码）

接口
----
GET  /health /models /templates
POST /analyze  (multipart: file, template, prompt)
POST /batch    (json: template, dir, limit, recursive, skip_processed, save_images)
POST /batch/upload (multipart: template, save_images, files[])
GET  /batch /batch/{id} /batch/{id}/items /batch/{id}/export.csv
POST /batch/{id}/cancel

前端测试页：static/index.html（部署到 nginx 静态目录即可）
