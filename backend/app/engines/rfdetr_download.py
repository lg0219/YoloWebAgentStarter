"""RF-DETR 权重的并行下载。

框架自带的下载是单连接，国内直连 repo.roboflow.com 只有十几 KB/s，349MB 要下几个小时。
这里改成多连接分段下载，实测 8 连接能从个位数 KB/s 拉到 500 KB/s 以上，同时上报百分比。
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_WORKERS = 8
USER_AGENT = "YWA/0.1"


def remote_size(url: str, timeout: int = 30) -> int:
    """用 Range 探测文件总大小。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range") or ""
        response.read(1)
    if "/" not in content_range:
        raise IOError("服务器未返回文件大小")
    return int(content_range.rsplit("/", 1)[1])


def _download_chunk(url: str, path: Path, start: int, end: int, counter, lock, stop: threading.Event) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "r+b") as handle:
        handle.seek(start)
        while not stop.is_set():
            data = response.read(256 * 1024)
            if not data:
                break
            handle.write(data)
            with lock:
                counter[0] += len(data)


def download(
    url: str,
    target: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    on_progress=None,
) -> Path:
    """并行分段下载到 target；失败会清理临时文件后抛出。"""
    total = remote_size(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with open(temporary, "wb") as handle:
        handle.truncate(total)

    counter = [0]
    lock = threading.Lock()
    stop = threading.Event()
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + CHUNK_SIZE, total) - 1
        ranges.append((start, end))
        start = end + 1

    def run(pairs: list[tuple[int, int]]) -> None:
        for begin, finish in pairs:
            if stop.is_set():
                return
            _download_chunk(url, temporary, begin, finish, counter, lock, stop)

    # 段数按连接数轮转，避免把相邻段全压在一个线程
    buckets: list[list[tuple[int, int]]] = [[] for _ in range(max(1, min(workers, len(ranges))))]
    for index, pair in enumerate(ranges):
        buckets[index % len(buckets)].append(pair)

    threads = [threading.Thread(target=run, args=(bucket,), daemon=True) for bucket in buckets if bucket]
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.5)
            if on_progress is not None:
                with lock:
                    done = counter[0]
                on_progress(done, total)
            if counter[0] >= total:
                break
        for thread in threads:
            thread.join()
    except Exception:
        stop.set()
        temporary.unlink(missing_ok=True)
        raise

    if temporary.stat().st_size != total:
        temporary.unlink(missing_ok=True)
        raise IOError("下载文件大小与服务器不一致")
    os.replace(temporary, target)
    if on_progress is not None:
        on_progress(total, total)
    return target
