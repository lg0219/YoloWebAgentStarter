# 内嵌的 RT-DETRv2（独立版）

## 来源

- 上游仓库：https://github.com/lyuwenyu/RT-DETR （`rtdetrv2_pytorch` 目录）
- 许可证：Apache-2.0，见同目录 `LICENSE`
- 内嵌方式：仅复制 `src/`、`configs/`、`tools/` 与 `requirements.txt`，不包含上游的 `references/`、`dataset/`、Docker 相关文件
- 目的：让「RT-DETR（独立版 v2）」作为可选训练引擎直接可用，用户不需要自己克隆仓库

## 本地补丁（升级上游时需要重新打）

1. `src/core/yaml_utils.py`：读取 yaml 时显式 `encoding='utf-8'`。原因是 Windows 默认按 GBK 解码，项目路径含中文时会直接报错。
2. `src/core/yaml_utils.py`：`__include__` 的绝对路径判断补上 `os.path.isabs`。原因是原实现只认 `/` 开头的路径，Windows 盘符路径会被误当成相对路径拼接。

## 升级做法

用上游 `rtdetrv2_pytorch` 覆盖 `src/`、`configs/`、`tools/`，然后按上面两条重新打补丁，并跑一遍引擎测试。
