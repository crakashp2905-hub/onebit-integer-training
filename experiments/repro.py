"""Reproducibility pinning.

Measured at M1: torch.linalg.qr is THREAD-COUNT DEPENDENT. Building the same
quadratic under 1 vs 4 torch threads gives matrices differing by ~7.7e-7
relative -- pure float32 roundoff, but the trajectory is not roundoff-stable.

Measured amplification of that 7.7e-7 input perturbation into final rel_gap
(kappa=100, 8 bits, 3000 steps):

    RTN     2.3e-7  relative change   (no amplification -- a freeze is a stable
                                       attractor and absorbs the perturbation)
    SR      3.4e-2                    ( ~44,000x amplification)
    EF      2.6e-1                    (~343,000x amplification)

Consequence for methodology: any effect smaller than a few tens of percent in
SR/EF runs is NOT resolved by seed spread alone, and results are only comparable
across machines if the thread count is pinned. Call pin() at the top of every
experiment and record threads() in every results row.
"""

from __future__ import annotations

import atexit
import os
import random
import sys

import torch

THREADS = int(os.environ.get("ONEBIT_THREADS", "4"))


def pin(threads: int | None = None, seed: int = 0) -> int:
    """Pin thread count and global seeds. Returns the thread count in force."""
    n = THREADS if threads is None else threads
    torch.set_num_threads(n)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    random.seed(seed)
    return n


def threads() -> int:
    return torch.get_num_threads()


def cpu_clock() -> float:
    """Process CPU time in seconds -- sum of user+system across threads.

    WALL CLOCK IS NOT USABLE HERE. This machine sleeps overnight, and a
    suspended process keeps accruing wall time while doing no work: measured
    runs of 738 and 1317 minutes for work that takes 10-25 minutes awake,
    with val_loss identical to their siblings. Throughput derived from wall
    clock is therefore meaningless on any run that spans a sleep.

    time.process_time() counts only CPU actually consumed, so it is unaffected.
    Note it sums over threads, so with N torch threads busy it advances up to N
    times faster than wall time; use it for run-to-run COMPARISON, and divide by
    the thread count for a wall-clock-equivalent estimate.
    """
    import time as _t

    return _t.process_time()


# --------------------------------------------------------------- sleep guard

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> bool:
    """Ask Windows to keep the system awake for as long as THIS PROCESS runs.

    Uses SetThreadExecutionState, the documented API a media player or an
    installer uses to say "I am doing work, do not suspend me". It is a runtime
    request scoped to the calling thread, not a configuration change: nothing is
    written to the power plan, and the request evaporates when the process
    exits or calls allow_sleep(). The user's settings are left exactly as found.

    This matters because a suspended process keeps accruing wall time while
    doing no work -- measured here as a 738-minute run of 10 minutes of actual
    compute -- which corrupts every throughput number and stretches a 15-hour
    job across days.

    Returns True if the request was accepted. No-ops on non-Windows.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        r = ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        return r != 0
    except Exception:
        return False


def allow_sleep() -> bool:
    """Release the keep-awake request. Called automatically at process exit."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS) != 0
    except Exception:
        return False
