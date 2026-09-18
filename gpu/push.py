"""Ship the repo to Kaggle as a PRIVATE dataset and run a PRIVATE GPU kernel.

  python gpu/push.py --configs R6_attn8,R7_head8 --steps 10000

Two uploads, in this order:

  1. a dataset containing the code and the FROZEN CORPUS. The corpus goes with
     the code on purpose: the methodology is "tokenize once, freeze, replay",
     and a GPU run that re-derives its own tokens is not comparable with the CPU
     runs it will be plotted against.
  2. a kernel whose entry point is gpu/entry.py, attached to that dataset.

Nothing is uploaded until the secret scan passes, and both artifacts are
private. This is not a policy gesture: the payload is assembled by globbing a
working tree, and a working tree is exactly where a stray token ends up.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG = "onebit-ladder-code"
KERNEL = "onebit-ladder-gpu"

#: what the dataset payload contains. Explicit, never "everything": results/
#: alone is tens of megabytes of CSV and PNG that the kernel has no use for.
INCLUDE = [
    "audit/**/*.py", "data/*.py", "experiments/*.py", "models/*.py",
    "optim/*.py", "quant/*.py", "gpu/entry.py",
    "data/cache/bpe_2048.json", "data/cache/tokens_2048.npz",
]

#: Substring/regex pairs. Deliberately noisy -- a false positive costs one
#: glance, a false negative costs a leaked credential.
SECRET_PATTERNS = [
    ("kaggle key", re.compile(r'"key"\s*:\s*"[0-9a-f]{20,}"')),
    ("aws", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai", re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"[Bb]earer\s+[A-Za-z0-9\-._~+/]{24,}")),
]


def collect() -> list[Path]:
    files: list[Path] = []
    for pat in INCLUDE:
        files.extend(sorted(p for p in ROOT.glob(pat) if p.is_file()))
    if not files:
        sys.exit("payload is empty -- INCLUDE matched nothing")
    return files


def scan(files: list[Path]) -> None:
    """Refuse to upload if anything smells like a credential."""
    hits = []
    for f in files:
        if f.suffix in {".npz", ".pt", ".png"}:
            continue                      # binary, and none of it is authored
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                try:
                    where = f.relative_to(ROOT)
                except ValueError:
                    where = f          # a file from outside the tree (tests)
                hits.append(f"{where}:{line}  [{label}]")
    if hits:
        print("SECRET SCAN FAILED -- nothing was uploaded:", file=sys.stderr)
        for h in hits:
            print("  " + h, file=sys.stderr)
        sys.exit(2)
    print(f"[scan] {len(files)} files, no secrets found")


def stage(files: list[Path], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for f in files:
        dst = out / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"[stage] {out}  {total/1e6:.1f} MB")


def cli_failed(returncode: int, stdout: str) -> bool:
    """The Kaggle CLI reports some rejections as TEXT with exit code 0 -- e.g.
    'Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.' Trusting
    the exit code alone printed 'pushed.' for a push that never happened, and the
    watcher then reported the previous session's COMPLETE as this one's."""
    return returncode != 0 or "error" in stdout.lower()


def kaggle(*args: str) -> None:
    print("+ kaggle " + " ".join(args))
    r = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       cwd=ROOT, text=True, capture_output=True)
    print(r.stdout.strip())
    if cli_failed(r.returncode, r.stdout):
        print(r.stderr.strip(), file=sys.stderr)
        print("[push] FAILED -- nothing is running on Kaggle.", file=sys.stderr)
        sys.exit(r.returncode or 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="kaggle username (else from ~/.kaggle)")
    ap.add_argument("--configs", default="", help="comma-separated rung names")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    user = args.user
    if not user:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        user = api.config_values["username"]
    print(f"[user] {user}")

    files = collect()
    scan(files)

    stage_dir = ROOT / "gpu" / "_payload"
    stage(files, stage_dir)
    (stage_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    kdir = ROOT / "gpu" / "_kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    # Kaggle uploads ONLY the code file named in kernel-metadata.json, so a
    # sidecar config file would silently not ship. Inject instead.
    entry = (ROOT / "gpu" / "entry.py").read_text(encoding="utf-8")
    entry = entry.replace('CONFIGS = ""', f'CONFIGS = {args.configs!r}')
    entry = entry.replace("STEPS = 10000", f"STEPS = {args.steps}")
    assert f"CONFIGS = {args.configs!r}" in entry, "config injection failed"
    (kdir / "entry.py").write_text(entry, encoding="utf-8")
    meta = {
        "id": f"{user}/{KERNEL}",
        "title": KERNEL,
        "code_file": "entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,          # NOT negotiable
        "enable_gpu": True,
        "enable_internet": False,    # the payload is self-contained; keep it so
        "dataset_sources": [f"{user}/{SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                               encoding="utf-8")
    if not meta["is_private"]:
        sys.exit("refusing to push a public kernel")

    if args.dry_run:
        print("[dry-run] nothing uploaded")
        return 0

    kaggle("datasets", "version", "-p", str(stage_dir), "-m",
           f"code + frozen corpus ({len(files)} files)", "--dir-mode", "zip")
    kaggle("kernels", "push", "-p", str(kdir))
    print(f"\npushed. watch with:  python gpu/watch.py --kernel {user}/{KERNEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
