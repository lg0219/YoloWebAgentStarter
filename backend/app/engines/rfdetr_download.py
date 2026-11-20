"""RF-DETR 权重的并行下载。

框架自带的下载是单连接，国内直连 repo.roboflow.com 只有十几 KB/s，349MB 要下几个小时。
这里改成多连接分段下载，实测 8 连接能从个位数 KB/s 拉到数百 KB/s，同时上报百分比。

注意：目标文件会预先占位成最终大小，所以「文件大小」不等于下载进度；判断完成看统计的
已下载字节数，以及下载结束后是否成功改名。
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_WORKERS = 8
USER_AGENT = "YWA/0.1"
RETRIES = 3


def remote_size(url: str, timeout: int = 30) -> int:
    """用 Range 探测文件总大小。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range") or ""
        response.read(1)
    if "/" not in content_range:
        raise IOError("服务器未返回文件大小")
    return int(content_range.rsplit("/", 1)[1])


def download(
    url: str,
    target: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    on_progress=None,
) -> Path:
    """并行分段下载到 target；全部段成功、校验大小一致后才改名为目标文件。"""
    total = remote_size(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with open(temporary, "wb") as handle:
        handle.truncate(total)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + CHUNK_SIZE, total) - 1
        ranges.append((start, end))
        start = end + 1

    lock = threading.Lock()
    stop = threading.Event()
    chunk_done: dict[int, int] = {index: 0 for index in range(len(ranges))}

    def progress(current: int) -> None:
        if on_progress is None:
            return
        try:
            on_progress(current, total)
        except Exception:
            pass

    def download_chunk(index: int, begin: int, finish: int) -> None:
        need = finish - begin + 1
        attempt = 0
        while attempt < RETRIES and not stop.is_set():
            attempt += 1
            written = 0
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={begin}-{finish}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "r+b") as handle:
                    handle.seek(begin)
                    while not stop.is_set() and written < need:
                        data = response.read(min(256 * 1024, need - written))
                        if not data:
                            break
                        handle.write(data)
                        written += len(data)
                        with lock:
                            chunk_done[index] = written
                with lock:
                    chunk_done[index] = written
                if written >= need:
                    return
            except Exception:
                if stop.is_set():
                    return
        # 重试用尽仍不完整，整体失败
        with lock:
            stop.set()

    def run(chunks: list[tuple[int, int, int]]) -> None:
        for index, begin, finish in chunks:
            if stop.is_set():
                return
            download_chunk(index, begin, finish)

    buckets: list[list[tuple[int, int, int]]] = [[] for _ in range(max(1, min(workers, len(ranges))))]
    for index, (begin, finish) in enumerate(ranges):
        buckets[index % len(buckets)].append((index, begin, finish))

    threads = [threading.Thread(target=run, args=(bucket,), daemon=True) for bucket in buckets if bucket]
    for thread in threads:
        thread.start()

    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.5)
            with lock:
                current = sum(chunk_done.values())
            progress(current)
        for thread in threads:
            thread.join()
    except Exception:
        stop.set()
        temporary.unlink(missing_ok=True)
        raise

    with lock:
        current = sum(chunk_done.values())
    if stop.is_set() or current != total:
        temporary.unlink(missing_ok=True)
        raise IOError(f"下载不完整：{current}/{total} 字节")
    os.replace(temporary, target)
    progress(total)
    return target
