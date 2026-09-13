"""Kaggle kernel entry point. Runs the ablation ladder on the attached GPU.

Only this file is uploaded as the kernel; everything else arrives as a mounted
dataset. Kaggle uploads the code file alone, so the run configuration is
INJECTED into the block below by gpu/push.py rather than shipped alongside.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------- INJECTED BY push.py
CONFIGS = ""
STEPS = 10000
# ------------------------------------------------------------------------------

OUT = Path("/kaggle/working")
CACHE = OUT / "cache"


def find_repo() -> Path:
    """Locate the mounted payload by SEARCHING, not by assuming a path.

    The documented-looking path /kaggle/input/<dataset>/ is wrong. The real one
    is /kaggle/input/datasets/<user>/<dataset>/, and that cost a whole failed
    session to discover. So: print the tree, then search it.
    """
    root = Path("/kaggle/input")
    print("=" * 70)
    print("mounted inputs:")
    n = 0
    for p in sorted(root.rglob("*")):
        print(" ", p)
        n += 1
        if n > 400:
            print("  ... (truncated)")
            break
    print("=" * 70)

    for marker in root.rglob("experiments/m5_ladder.py"):
        repo = marker.parent.parent
        print(f"[repo] found at {repo}")
        return repo
    sys.exit("could not find experiments/m5_ladder.py anywhere under /kaggle/input")


def main() -> int:
    repo = find_repo()
    CACHE.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # The dataset mount is READ-ONLY. Without these the tokenizer cache write
    # dies with OSError: Read-only file system, several minutes into the job.
    env["ONEBIT_CACHE"] = str(CACHE)
    env["ONEBIT_RESULTS"] = str(OUT)
    env["PYTHONPATH"] = str(repo)

    # the frozen corpus ships WITH the code, so copy it where the cache lives
    src = repo / "data" / "cache"
    if src.exists():
        for f in src.iterdir():
            if f.is_file():
                (CACHE / f.name).write_bytes(f.read_bytes())
        print(f"[cache] seeded {len(list(CACHE.iterdir()))} frozen-corpus files")

    import torch
    print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    cmd = [sys.executable, str(repo / "experiments" / "m5_ladder.py"),
           "--steps", str(STEPS)]
    if CONFIGS:
        cmd += ["--only", CONFIGS]
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo), env=env)

    for f in sorted(OUT.glob("*.csv")):
        print(f"[out] {f.name}  {f.stat().st_size/1e3:.1f} kB")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
