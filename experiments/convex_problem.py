"""A convex quadratic with everything known in closed form.

    f(w) = 0.5 * w^T A w - b^T w

Constructed so that the optimum is known exactly BY CONSTRUCTION rather than by
solving a linear system: pick w*, then set b = A w*. Then

    grad f(w) = A w - b = A (w - w*)
    f(w) - f*  = 0.5 * (w - w*)^T A (w - w*)          always >= 0, no cancellation

A has eigenvalues log-spaced in [1, kappa], so the condition number is exactly
kappa and the optimal gradient-descent step is lr = 2/(L+mu) = 2/(kappa+1) with
convergence rate rho = (kappa-1)/(kappa+1) per step in the A-norm.

Why a quadratic: the gradient is known exactly at every point, so the predicted
stall condition ||g||_inf < D/(2*lr) can be CHECKED rather than assumed. A real
model gives no such ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Quadratic:
    A: torch.Tensor
    b: torch.Tensor
    w_star: torch.Tensor
    kappa: float
    d: int
    spectrum: str = "logspace"

    def grad(self, w: torch.Tensor) -> torch.Tensor:
        return self.A @ w - self.b

    def gap(self, w: torch.Tensor) -> float:
        """f(w) - f*, computed from the stable difference form in float64."""
        dv = (w - self.w_star).double()
        return float(0.5 * dv @ (self.A.double() @ dv))

    def rel_gap(self, w: torch.Tensor, w0: torch.Tensor) -> float:
        g0 = self.gap(w0)
        return self.gap(w) / g0 if g0 > 0 else 0.0

    @property
    def lr_opt(self) -> float:
        return 2.0 / (self.kappa + 1.0)

    @property
    def rate_opt(self) -> float:
        """Theoretical per-step contraction factor at the optimal step size."""
        return (self.kappa - 1.0) / (self.kappa + 1.0)


def make_spectrum(d: int, kappa: float, kind: str = "logspace", p: float = 4.0) -> torch.Tensor:
    """Eigenvalue spectra spanning [1, kappa].

    The M1 results were measured on "logspace" only, and the SR-vs-EF behaviour
    plausibly depends on the spectrum. Real neural-network Hessians look nothing
    like log-spaced: they are dominated by a near-zero bulk with a small number
    of large outliers (the "bulk + outliers" picture). So the ladder needs to
    know whether the M1 findings are spectrum-dependent before they are trusted
    as guidance for R5.

      logspace       log-spaced in [1, kappa] -- the M1 default
      powerlaw       lam = 1 + (kappa-1) * u^p, u linear in [1,0]. p>1 pushes
                     most mass toward the SOFT end.
      bulk_outlier   95% of eigenvalues exactly 1, 5% log-spaced up to kappa.
                     Closest to a measured NN Hessian.
    """
    if kappa == 1.0:
        return torch.ones(d)
    if kind == "logspace":
        return torch.logspace(0.0, float(torch.log10(torch.tensor(kappa))), d)
    if kind == "powerlaw":
        u = torch.linspace(1.0, 0.0, d)
        return 1.0 + (kappa - 1.0) * u.pow(p)
    if kind == "bulk_outlier":
        n_out = max(1, d // 20)
        lam = torch.ones(d)
        lam[:n_out] = torch.logspace(0.0, float(torch.log10(torch.tensor(kappa))), n_out)
        return lam.flip(0)
    raise ValueError(f"unknown spectrum {kind!r}")


def make_quadratic(
    d: int = 1000,
    kappa: float = 100.0,
    seed: int = 0,
    spectrum: str = "logspace",
    p: float = 4.0,
) -> Quadratic:
    g = torch.Generator().manual_seed(seed)
    lam = make_spectrum(d, kappa, spectrum, p)
    M = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(M)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)  # kill asymmetry from float roundoff
    w_star = torch.randn(d, generator=g)
    b = A @ w_star
    return Quadratic(A=A, b=b, w_star=w_star, kappa=float(kappa), d=d, spectrum=spectrum)


def verify_problem(p: Quadratic, tol: float = 1e-3) -> dict[str, float]:
    """Cross-check the construction against its own closed form.

    Returns a dict of measured residuals; the caller asserts on them. Nothing
    here is assumed -- every claim in the docstring is measured.
    """
    eig = torch.linalg.eigvalsh(p.A.double())
    grad_at_opt = p.grad(p.w_star).abs().max().item()
    scale = p.b.abs().max().item()
    # f(w*) computed the long way must match the closed form -0.5 w*^T A w*
    ws = p.w_star.double()
    f_long = float(0.5 * ws @ (p.A.double() @ ws) - p.b.double() @ ws)
    f_closed = float(-0.5 * ws @ (p.A.double() @ ws))
    return {
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "kappa_measured": float(eig.max() / eig.min()),
        "kappa_declared": p.kappa,
        "grad_at_optimum_rel": grad_at_opt / max(scale, 1e-30),
        "f_star_mismatch": abs(f_long - f_closed) / max(abs(f_closed), 1e-30),
        "symmetry_err": float((p.A - p.A.T).abs().max()),
        "gap_at_optimum": p.gap(p.w_star),
    }
