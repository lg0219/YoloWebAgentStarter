// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingView } from "./TrainingView";
import type { Dataset } from "../types";

vi.mock("../api/client", () => ({
  api: {
    listTrainingTasks: vi.fn().mockResolvedValue({ items: [] }),
    listTrainingDevices: vi.fn().mockResolvedValue({ items: [] }),
    createTrainingTask: vi.fn(),
    stopTrainingTask: vi.fn(),
    resumeTrainingTask: vi.fn(),
    getTrainingLogs: vi.fn().mockResolvedValue({ task_id: "", logs: "", line_count: 0 }),
    getTrainingSummary: vi.fn().mockResolvedValue({
      task_id: "",
      status: "pending",
      training_config: {},
      dataset: {},
      progress: { epoch: 0, total_epochs: 0, percent: 0 },
      metrics: {},
      checkpoints: {},
      log_summary: { line_count: 0, tail: [] },
      risks: [],
      next_steps: [],
    }),
    downloadCheckpointUrl: vi.fn().mockReturnValue("#"),
    autoSplitImages: vi.fn(),
  },
}));

const detect: Dataset = { id: "ds-detect", name: "detect-set", description: null, task_type: "detect", image_count: 4, annotated_image_count: 4, class_count: 1, created_at: "", updated_at: "" };
const segment: Dataset = { id: "ds-segment", name: "segment-set", description: null, task_type: "segment", image_count: 2, annotated_image_count: 2, class_count: 1, created_at: "", updated_at: "" };

let root: Root | undefined;
let container: HTMLDivElement | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => { container = document.createElement("div"); document.body.append(container); root = createRoot(container); });
afterEach(() => { act(() => root?.unmount()); container?.remove(); });

describe("TrainingView field help", () => {
  // 2026-11-05 hunter 验收反馈：帮助气泡只能由「?」触发，不能整行可点
  it("only the ? icon toggles field help; the label row is not a click target", async () => {
    await act(async () => { root?.render(<TrainingView datasets={[detect]} dataset={detect} onDatasetChange={vi.fn()} />); });
    const labels = Array.from(container?.querySelectorAll<HTMLLabelElement>(".training-ws-field-label label") ?? []);
    const epochsLabel = labels.find((item) => item.textContent === "Epochs");
    expect(epochsLabel).toBeDefined();

    // 结构断言：字段容器不再是 <label>（整行可点），? 与 label 是兄弟节点
    const field = epochsLabel?.closest(".training-ws-field");
    expect(field?.tagName).toBe("DIV");
    const helpButton = field?.querySelector<HTMLButtonElement>(".training-field-help-button");
    expect(helpButton).toBeDefined();

    // 点 label 文字：不弹气泡
    await act(async () => { epochsLabel?.click(); });
    expect(container?.querySelector(".training-field-help-bubble")).toBeNull();

    // 点 ?：弹出气泡且内容属该字段
    await act(async () => { helpButton?.click(); });
    const bubble = container?.querySelector(".training-field-help-bubble");
    expect(bubble?.textContent).toContain("所有图片");

    // 再点一次：收起
    await act(async () => { helpButton?.click(); });
    expect(container?.querySelector(".training-field-help-bubble")).toBeNull();
  });
});

describe("TrainingView dataset selection", () => {
  it("allows selecting the dataset used for a new training task", async () => {
    const onDatasetChange = vi.fn();
    await act(async () => { root?.render(<TrainingView datasets={[detect, segment]} dataset={detect} onDatasetChange={onDatasetChange} />); });
    const select = Array.from(container?.querySelectorAll<HTMLSelectElement>("select") ?? []).find((item) => item.value === detect.id);
    expect(select).toBeDefined();
    await act(async () => { if (select) { select.value = segment.id; select.dispatchEvent(new Event("change", { bubbles: true })); } });
    expect(onDatasetChange).toHaveBeenCalledWith(segment);
  });

  it("renders upstream-style workspace sections", async () => {
    await act(async () => { root?.render(<TrainingView datasets={[detect]} dataset={detect} onDatasetChange={vi.fn()} />); });
    expect(container?.querySelector("#training-config-anchor")).toBeTruthy();
    expect(container?.querySelector("#experiment-history")).toBeTruthy();
    expect(container?.querySelector(".training-dataset-rail")).toBeTruthy();
    expect(container?.textContent).toContain("开始训练");
  });
});
