"""Wait for several Kaggle notebooks (also ones not pushed yet), download each output once it finishes,
then refresh experiments/. Robust to transient API failures.

    uv run python -u tools/fetch_when_done.py <slug> [<slug> ...]
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from bench_queue import kaggle, log, status  # noqa: E402
from overnight import build_v1, refresh  # noqa: E402

DONE = ("complete", "error", "cancelacknowledged", "cancelled")
USER = "longquanton"


def fetch(slug: str) -> None:
    # "slug" (our account) or "owner/slug" (a teammate's notebook shared with us)
    kernel = slug if "/" in slug else f"{USER}/{slug}"
    slug = kernel.split("/")[1]
    tag = slug.removeprefix("rmit-sgs-gendata-").removesuffix("-c1")
    seen_running, last = False, ""
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        s = status(kernel)
        if s != last:
            log(f"[{tag}] status {s}")
            last = s
        seen_running |= s in ("running", "queued")
        if s in DONE and seen_running:
            break
        time.sleep(60)
    dest = ROOT / "outputs" / slug
    dest.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", kernel, "-p", str(dest))
    log(f"[{tag}] output -> outputs/{slug} ({'ok' if r.returncode == 0 else r.stderr[-200:]})")


def main() -> None:
    threads = [threading.Thread(target=fetch, args=(s,)) for s in sys.argv[1:]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        refresh()
        build_v1("mmbert-base")
    log("fetch_when_done finished")


if __name__ == "__main__":
    main()
