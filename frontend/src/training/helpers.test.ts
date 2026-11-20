import { describe, expect, it } from "vitest";
import { applyTaskToForm, defaultForm, modelChoicesFor } from "./helpers";
import type { TrainingTask } from "../types";

describe("training model choices", () => {
  it("offers the supported YOLO26, YOLO11 and YOLOv8 families for detection", () => {
    const values = modelChoicesFor("detect").map((item) => item.value);

    expect(values).toEqual(expect.arrayContaining([
      "yolo26n.pt",
      "yolo11n.pt",
      "yolo11m.pt",
      "yolov8n.pt",
    ]));
    expect(values).not.toEqual(expect.arrayContaining(["yolo13n.pt"]));
  });

  it("keeps task-specific suffixes for segmentation, OBB and classification", () => {
    expect(modelChoicesFor("segment").map((item) => item.value)).toContain("yolo26n-seg.pt");
    expect(modelChoicesFor("classify").map((item) => item.value)).toContain("yolov8n-cls.pt");
  });

  it("offers the Ultralytics RT-DETR checkpoints for detection only", () => {
    const detect = modelChoicesFor("detect");
    const values = detect.map((item) => item.value);

    expect(values).toEqual(expect.arrayContaining(["rtdetr-l.pt", "rtdetr-x.pt"]));
    expect(detect.find((item) => item.value === "rtdetr-l.pt")?.label).toBe("RT-DETR-L（检测） · rtdetr-l.pt");

    for (const task of ["segment", "obb", "classify"] as const) {
      expect(modelChoicesFor(task).map((item) => item.value).some((value) => value.startsWith("rtdetr"))).toBe(false);
    }
  });
});

describe("deterministic training switch", () => {
  const task = {
    id: "train-1",
    name: "demo",
    task_type: "detect",
    device: "cpu",
    model_name: "yolo11n.pt",
    epochs: 3,
    batch_size: 2,
    img_size: 640,
    workers: 0,
    seed: 42,
    val_ratio: 0.2,
    patience: null,
    optimizer: null,
    lr0: null,
    deterministic: false,
  } as TrainingTask;

  it("defaults to on, matching the Ultralytics default", () => {
    expect(defaultForm("detect").deterministic).toBe(true);
  });

  it("keeps a retried task's stored value", () => {
    expect(applyTaskToForm(task).deterministic).toBe(false);
    expect(applyTaskToForm({ ...task, deterministic: true }).deterministic).toBe(true);
  });
});
