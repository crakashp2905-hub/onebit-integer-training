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


def normalize(s: str) -> str:
    """The API returns 'KernelWorkerStatus.RUNNING', not 'running'. Normalizing
    inline in the poll loop left that untested, which is precisely how the
    original false-COMPLETE bug got in: the classification of a status string is
    the part that decides whether a result gets reported, so it is the part that
    needs a test."""
    return s.strip().lower().split(".")[-1].replace("_", "")


def is_terminal(s: str) -> bool:
    return normalize(s) in TERMINAL


def accept_terminal(s: str, seen_active: bool) -> bool:
    """A terminal status counts only after this watcher has seen the job ACTIVE.

    Right after a push, the API still reports the PREVIOUS version's final
    state. Without this guard the watcher read a stale COMPLETE and declared a
    job done after 0.00 h -- a false completion by a new route."""
    return is_terminal(s) and seen_active


def status(api, ref: str) -> tuple[str, str]:
    # kernels_status takes the full "user/slug" as ONE argument in the current
    # client, and took two in an older one. Try both rather than pin a version:
    # a signature mismatch here is indistinguishable, to the retry loop, from a
    # network fault, so it would otherwise retry forever without ever saying why.
    try:
        r = api.kernels_status(ref)
    except TypeError:
        r = api.kernels_status(*ref.split("/", 1))
    s = getattr(r, "status", None) or (r.get("status") if isinstance(r, dict) else None)
    msg = getattr(r, "failureMessage", None) or (
        r.get("failureMessage") if isinstance(r, dict) else None)
    return str(s or "unknown"), str(msg or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None, help="user/kernel-slug")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--stale-grace-min", type=float, default=30.0,
                    help="how long a terminal status may persist before the "
                         "job is ever seen active; after this, report STALE")
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
    seen_active = False
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
        if not is_terminal(s):
            seen_active = True
        elif not accept_terminal(s, seen_active):
            if (time.time() - t0) / 60 > args.stale_grace_min:
                print()
                print(f"[watch] STALE: status has been {s} since the watch began "
                      f"and the job was never seen running. The push probably "
                      f"did not take. THIS IS NOT A COMPLETION.")
                return 4
            time.sleep(args.interval)
            continue
        if accept_terminal(s, seen_active):
            el = (time.time() - t0) / 3600
            print(f"\n[watch] terminal: {s} after {el:.2f} h")
            if msg:
                print(f"[watch] {msg}")
            return 0 if normalize(s) == "complete" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
