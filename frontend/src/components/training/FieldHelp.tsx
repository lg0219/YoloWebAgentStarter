import { useEffect, useRef, useState } from "react";

interface FieldHelpProps {
  /** 气泡内展示的中文说明文案 */
  text: string;
  /** 「?」按钮的无障碍名称 */
  title?: string;
}

/**
 * 训练配置字段旁的「?」帮助气泡（2026-11-05 新增）。
 * 点击「?」图标切换显示；Esc 键或点击组件外部任意处关闭。
 * 只做中文文案，跟随训练面板使用，不做 i18n。
 */
export function FieldHelp({ text, title = "查看说明" }: FieldHelpProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && rootRef.current && !rootRef.current.contains(target)) setOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  return (
    <span className="training-field-help" ref={rootRef}>
      <button
        type="button"
        className="training-field-help-button"
        aria-label={title}
        aria-expanded={open}
        onClick={(event) => {
          // 阻止冒泡到外层 <label>：否则点击「?」会顺带聚焦输入框
          event.preventDefault();
          event.stopPropagation();
          setOpen((value) => !value);
        }}
      >
        ?
      </button>
      {open && (
        <span className="training-field-help-bubble" role="note">
          {text}
        </span>
      )}
    </span>
  );
}
