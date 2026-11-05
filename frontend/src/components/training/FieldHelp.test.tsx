// @vitest-environment jsdom

import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FieldHelp } from "./FieldHelp";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function render(ui: ReactElement) {
  act(() => {
    root.render(ui);
  });
}

function helpButton(): HTMLButtonElement {
  const button = container.querySelector<HTMLButtonElement>("button.training-field-help-button");
  if (!button) throw new Error("help button not rendered");
  return button;
}

function bubble(): HTMLElement | null {
  return container.querySelector(".training-field-help-bubble");
}

describe("FieldHelp", () => {
  it("默认只渲染 ? 按钮，不展示说明", () => {
    render(<FieldHelp text="训练的遍数" />);
    const button = helpButton();
    expect(button.getAttribute("aria-label")).toBe("查看说明");
    expect(button.getAttribute("aria-expanded")).toBe("false");
    expect(bubble()).toBeNull();
  });

  it("点击 ? 展开说明，再点一次收起", () => {
    render(<FieldHelp text="每次迭代处理的图片数" />);
    const button = helpButton();

    act(() => button.click());
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(bubble()?.textContent).toBe("每次迭代处理的图片数");

    act(() => button.click());
    expect(button.getAttribute("aria-expanded")).toBe("false");
    expect(bubble()).toBeNull();
  });

  it("展开后按 Esc 关闭", () => {
    render(<FieldHelp text="初始学习率" />);
    act(() => helpButton().click());
    expect(bubble()?.textContent).toBe("初始学习率");

    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(bubble()).toBeNull();
  });

  it("展开后点击组件外部关闭", () => {
    render(<FieldHelp text="数据加载进程数" />);
    act(() => helpButton().click());
    expect(bubble()?.textContent).toBe("数据加载进程数");

    act(() => {
      document.body.dispatchEvent(new Event("pointerdown", { bubbles: true }));
    });
    expect(bubble()).toBeNull();
  });
});
