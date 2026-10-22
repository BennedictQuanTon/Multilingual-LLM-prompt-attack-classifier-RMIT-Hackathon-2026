"""Push already-built notebooks one by one as soon as Kaggle accepts them (max 2 concurrent GPU sessions per
account), then wait for each and download its output.

    uv run python -u tools/push_when_free.py <build-dir>:<timeout-seconds> [...]
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from bench_queue import kaggle, log  # noqa: E402
from overnight import wait_and_fetch  # noqa: E402


def push_until_accepted(build: Path, timeout: int) -> str:
    kernel = json.loads((build / "kernel-metadata.json").read_text())["id"]
    tag = build.name.removeprefix("rmit-sgs-gendata-").removesuffix("-c1")
    while True:
        r = kaggle("kernels", "push", "-p", str(build), "-t", str(timeout))
        text = (r.stdout + r.stderr).strip()
        if r.returncode == 0 and "error" not in text.lower():
            log(f"[{tag}] pushed {kernel} (timeout {timeout}s)")
            return kernel
        if "Maximum batch GPU session count" not in text:
            log(f"[{tag}] push failed: {text[-200:]} — retrying")
        time.sleep(90)


def main() -> None:
    threads = []
    for spec in sys.argv[1:]:
        path, timeout = spec.rsplit(":", 1)
        build = ROOT / path
        tag = build.name.removeprefix("rmit-sgs-gendata-").removesuffix("-c1")
        kernel = push_until_accepted(build, int(timeout))
        t = threading.Thread(target=wait_and_fetch, args=(tag, kernel))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log("push_when_free finished")


if __name__ == "__main__":
    main()
