"""Global registry of quantization sites.

Purpose: quantization fails SILENTLY. A quantizer that returns its input, a spec
that never got wired in, a flag that defaults to off -- none of these raise, and
all of them produce a plausible-looking loss curve that is secretly the FP32
baseline. That is the single most likely way this project dies.

So the no-op check is a RUNTIME invariant, asserted after a warm-up step of every
run, not only a unit test. It is cheap: once a site has been observed to change
its input, the comparison is skipped forever after.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor

from .config import QuantSpec


@dataclass
class SiteStats:
    name: str
    spec: QuantSpec
    calls: int = 0
    changed_ever: bool = False
    zero_frac_sum: float = 0.0
    sat_frac_sum: float = 0.0
    scale_min: float = float("inf")
    scale_max: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def zero_frac_mean(self) -> float:
        return self.zero_frac_sum / self.calls if self.calls else 0.0

    @property
    def sat_frac_mean(self) -> float:
        return self.sat_frac_sum / self.calls if self.calls else 0.0


_SITES: dict[str, SiteStats] = {}


def register_site(name: str, spec: QuantSpec) -> None:
    if name in _SITES and _SITES[name].spec != spec:
        raise ValueError(
            "site " + name + " already registered with a different spec: "
            + _SITES[name].spec.describe() + " vs " + spec.describe()
        )
    _SITES.setdefault(name, SiteStats(name=name, spec=spec))


def record(name: str, x_in: Tensor, result) -> None:
    """Record one quantizer call. Auto-registers unseen sites."""
    st = _SITES.get(name)
    if st is None:
        register_site(name, result.spec)
        st = _SITES[name]
    st.calls += 1
    if not st.changed_ever and result.spec.kind != "none":
        # short-circuits permanently once True, so this costs one pass per site
        st.changed_ever = bool((result.q != x_in).any().item())
    s = result.stats
    st.zero_frac_sum += s.get("zero_frac", 0.0)
    st.sat_frac_sum += s.get("sat_frac", 0.0)
    if "scale_min" in s:
        st.scale_min = min(st.scale_min, s["scale_min"])
        st.scale_max = max(st.scale_max, s["scale_max"])


def assert_no_noops() -> None:
    """Raise if any active quantization site never changed its input.

    kind='none' sites are exempt: they are pass-through by design and serve as
    ablation controls.
    """
    offenders = sorted(
        s.name
        for s in _SITES.values()
        if s.spec.kind != "none" and s.calls > 0 and not s.changed_ever
    )
    if offenders:
        raise AssertionError(
            "quantization sites that never changed their input (silent no-op): "
            + ", ".join(offenders)
            + "\nThe quantizer is not wired in, or the spec is a no-op for this data."
        )
    inactive = sorted(s.name for s in _SITES.values() if s.calls == 0)
    if inactive:
        raise AssertionError(
            "registered quantization sites that were never called: " + ", ".join(inactive)
        )


def dump_stats() -> list[dict[str, Any]]:
    rows = []
    for s in sorted(_SITES.values(), key=lambda v: v.name):
        if s.spec.kind == "int":
            bits: Any = s.spec.bits
        elif s.spec.kind == "ternary":
            bits = 1.58
        else:
            bits = ""
        rows.append(
            {
                "site": s.name,
                "spec": s.spec.describe(),
                "kind": s.spec.kind,
                "bits": bits,
                "calls": s.calls,
                "changed_ever": s.changed_ever,
                "zero_frac_mean": round(s.zero_frac_mean, 6),
                "sat_frac_mean": round(s.sat_frac_mean, 6),
                "scale_min": s.scale_min if s.scale_min != float("inf") else "",
                "scale_max": s.scale_max,
            }
        )
    return rows


def write_csv(path: str | Path) -> Path:
    rows = dump_stats()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def reset() -> None:
    _SITES.clear()


def sites() -> dict[str, SiteStats]:
    return _SITES
