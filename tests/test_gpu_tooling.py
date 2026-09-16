"""Tests for the Kaggle tooling.

Both tests here exist because the thing they check already failed once in a way
that corrupted a REPORTED RESULT rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpu import push, watch  # noqa: E402


# ------------------------------------------------------------------ the watcher

class _FlakyApi:
    """Fails, then returns garbage, then completes. None of the first two may
    be mistaken for a completion."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def kernels_status(self, *_):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"status": item, "failureMessage": None}


@pytest.mark.parametrize("bogus", [
    ConnectionError("Temporary failure in name resolution"),
    TimeoutError("read timed out"),
    RuntimeError("HTTP 503"),
])
def test_a_network_blip_is_never_a_completion(bogus):
    """The bug this replaces: a shell `case` whose default branch meant
    'finished', so one DNS failure reported a running job as complete."""
    api = _FlakyApi([bogus])
    with pytest.raises(type(bogus)):
        watch.status(api, "u/k")


@pytest.mark.parametrize("s", [
    "complete", "COMPLETE", "KernelWorkerStatus.COMPLETE",
    "error", "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCEL_ACKNOWLEDGED",
])
def test_real_terminal_states_are_recognised(s):
    """The API returns the enum's repr, not a bare word. A watcher that fails to
    recognise COMPLETE polls a finished job until it times out."""
    assert watch.is_terminal(s)


@pytest.mark.parametrize("s", [
    "running", "queued", "unknown", "", "pending",
    "KernelWorkerStatus.RUNNING", "KernelWorkerStatus.QUEUED",
    "SomethingNobodyHasSeenBefore",
])
def test_nothing_else_is_terminal(s):
    """Including the status string nobody has seen before -- an unrecognised
    status is a reason to keep watching, never a reason to report completion."""
    assert not watch.is_terminal(s)


# ------------------------------------------------------------------- the payload

def test_secret_scan_catches_a_planted_credential(tmp_path):
    """The payload is assembled by globbing a working tree, which is exactly
    where a stray token ends up."""
    f = tmp_path / "leak.py"
    f.write_text('TOKEN = "ghp_' + "a" * 36 + '"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        push.scan([f])
    assert e.value.code == 2


def test_secret_scan_passes_the_real_payload():
    """Not decorative: this is the assertion that actually runs before a push."""
    push.scan(push.collect())


def test_the_kernel_payload_can_never_be_pushed_public():
    src = (ROOT / "gpu" / "push.py").read_text(encoding="utf-8")
    assert '"is_private": True' in src
    assert "refusing to push a public kernel" in src
