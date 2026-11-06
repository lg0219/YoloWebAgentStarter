/**
 * 顶部品牌区显示偏好（2026-11-06 新增）。
 *
 * 与语言偏好同一机制：只存本机浏览器 localStorage，不写后端。
 * 未写入过时默认显示（true），保持既有用户的开箱体验不变。
 */
const STORAGE_KEY = "ywa.showBrandBar";

export function readBrandBar(): boolean {
  return window.localStorage.getItem(STORAGE_KEY) !== "false";
}

export function saveBrandBar(show: boolean): void {
  window.localStorage.setItem(STORAGE_KEY, show ? "true" : "false");
}
