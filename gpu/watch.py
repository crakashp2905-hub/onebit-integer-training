"""Poll a Kaggle kernel until it really finishes.

THE RULE THIS FILE EXISTS TO ENFORCE: only COMPLETE, ERROR and CANCEL_ACKNOWLEDGED
end the loop. Everything else -- an unrecognised status string, a timeout, a DNS
failure, an HTTP 5xx, any exception at all -- is transient and gets retried.

The previous version was a shell loop with a `case` statement whose default
branch meant "finished". A single DNS blip made it report a running job as
complete, and the run was reported as finished to the user on the strength of a
network hiccup. A watcher whose failure mode is a false COMPLETE is worse than
no watcher.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled", "canceled"}


def status(api, ref: str) -> tuple[str, str]:
    r = api.kernels_status(*ref.split("/", 1))
    s = getattr(r, "status", None) or (r.get("status") if isinstance(r, dict) else None)
    msg = getattr(r, "failureMessage", None) or (
        r.get("failureMessage") if isinstance(r, dict) else None)
    return str(s or "unknown"), str(msg or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None, help="user/kernel-slug")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=13.0,
                    help="Kaggle caps a session at 12h; this is the giving-up point")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    ref = args.kernel or f"{api.config_values['username']}/onebit-ladder-gpu"
    print(f"[watch] {ref}  every {args.interval}s")

    t0 = time.time()
    consecutive_errors = 0
    last = None
    while True:
        if (time.time() - t0) / 3600 > args.max_hours:
            print(f"[watch] gave up after {args.max_hours}h -- status never became "
                  f"terminal. THIS IS NOT A COMPLETION.")
            return 3
        try:
            s, msg = status(api, ref)
            consecutive_errors = 0
        except Exception as e:                  # noqa: BLE001 -- all of them
            # transient BY CONSTRUCTION. Never terminal, never a completion.
            consecutive_errors += 1
            print(f"  {datetime.now():%H:%M:%S}  poll failed ({consecutive_errors}): "
                  f"{type(e).__name__}: {e}", flush=True)
            time.sleep(min(args.interval * consecutive_errors, 900))
            continue

        if s != last:
            print(f"  {datetime.now():%H:%M:%S}  {s}"
                  + (f"  -- {msg}" if msg else ""), flush=True)
            last = s
        key = s.lower().replace("_", "").replace("kernelworkerstatus.", "")
        if key in TERMINAL:
            el = (time.time() - t0) / 3600
            print(f"\n[watch] terminal: {s} after {el:.2f} h")
            if msg:
                print(f"[watch] {msg}")
            return 0 if key == "complete" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
