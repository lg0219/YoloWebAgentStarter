import { describe, expect, it } from "vitest";
import { estimateMinutes, estimateOptimizerSteps, estimateTrainImages } from "./helpers";

describe("training estimate", () => {
  it("折算训练张数时扣掉验证集与未标注图", () => {
    // 数据集 25 张，其中 24 张已标注，val 比例 0.2 -> 约 19 张参与训练
    expect(estimateTrainImages(25, 24, 0.2)).toBe(19);
    // 全标注时用总量折算
    expect(estimateTrainImages(10, 10, 0.2)).toBe(8);
    // 没有标注信息时退回总量
    expect(estimateTrainImages(10, 0, 0.2)).toBe(8);
  });

  it("步数按训练张数算，而不是数据集总张数", () => {
    // 之前用 25 张会算成 70 步；实际训练 19 张、批 4 -> 5 步/轮 × 10
    expect(estimateOptimizerSteps(10, 19, 4)).toBe(50);
    expect(estimateOptimizerSteps(1, 3, 4)).toBe(1);
  });

  it("RT-DETR 在 CPU 上的预估接近实测（约 10 分钟）", () => {
    const minutes = estimateMinutes({
      engine: "rtdetrv2",
      epochs: 10,
      trainImages: 20,
      batchSize: 4,
      deviceType: "cpu",
      gpuCount: 1,
    });
    expect(minutes).not.toBeNull();
    expect(minutes!).toBeGreaterThanOrEqual(8);
    expect(minutes!).toBeLessThanOrEqual(15);
  });

  it("显卡上未标定的引擎不给数字，避免给出误导性估算", () => {
    expect(
      estimateMinutes({
        engine: "rtdetrv2",
        epochs: 10,
        trainImages: 20,
        batchSize: 4,
        deviceType: "cuda",
        gpuCount: 1,
      }),
    ).toBeNull();
    // YOLO 有历史经验值，显卡仍然给估算
    expect(
      estimateMinutes({
        engine: "yolo",
        epochs: 10,
        trainImages: 20,
        batchSize: 4,
        deviceType: "cuda",
        gpuCount: 1,
      }),
    ).not.toBeNull();
  });
});
