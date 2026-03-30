"""
TOST (Two One-Sided Tests) + Welch's t-test from summary statistics.
Generates console analysis *and* LaTeX tables with significance subcolumns.

Significance subcolumns (RED metric, RHS vs each baseline):
  Welch t-test stars  :  *** p<.001  ** p<.01  * p<.05  ~ p<.1  ns
  TOST dagger (†)     :  90 % CI ⊆ (−ε, +ε)  and  TOST p < 0.05

Usage
-----
    python tost_analysis.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MethodStats:
    """Summary statistics (mean ± std) for one method at one sample size."""

    red_mean: float
    red_std: float
    cpu_mean: float
    cpu_std: float


# Shorthand used in the DATA section below
MS = MethodStats


@dataclass
class TestResult:
    """Combined result from a Welch t-test and TOST equivalence test."""

    diff: float  # m1 − m2
    ci_lo: float  # lower bound of the (1 − 2α) CI
    ci_hi: float  # upper bound of the (1 − 2α) CI
    p_welch: float  # two-sided Welch p-value
    p_tost: float  # TOST p-value  (max of the two one-sided p-values)
    equivalent: bool  # True iff 90 % CI ⊆ (−ε, +ε)

    @property
    def stars(self) -> str:
        """Plain-text significance stars (for console output)."""
        return _sig_stars(self.p_welch)

    def latex_sig(self) -> str:
        """
        LaTeX-formatted cell for a significance column — AISTATS/ML convention.

        Priority (mutually exclusive, in order):
          1. TOST equivalence                       →  $\\approx$
          2. Welch p < 0.05  and  diff > 0  (win)   →  $\\checkmark$
          3. Welch p < 0.05  and  diff < 0  (loss)  →  $\\times$
          4. Otherwise (inconclusive)                →  --
        """
        if self.equivalent:
            return r"$\approx$"
        if self.p_welch < 0.05:
            return r"$\checkmark$" if self.diff > 0 else r"$\times$"
        return "--"


# ══════════════════════════════════════════════════════════════════════════════
# Statistical Functions
# ══════════════════════════════════════════════════════════════════════════════


def _welch_se_df(s1: float, s2: float, n1: int, n2: int) -> tuple[float, float]:
    """Return (standard error, Welch–Satterthwaite df) for a difference."""
    v1, v2 = s1**2 / n1, s2**2 / n2
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return se, df


def welch_ttest(
    m1: float,
    s1: float,
    m2: float,
    s2: float,
    n1: int,
    n2: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """
    Welch two-sample t-test from summary statistics.

    Returns
    -------
    diff    : m1 − m2
    df      : Welch–Satterthwaite degrees of freedom
    p_welch : two-sided p-value
    se      : standard error of the difference
    """
    n2 = n2 or n1
    diff = m1 - m2
    se, df = _welch_se_df(s1, s2, n1, n2)
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df)
    return diff, df, p, se


def run_tost(
    m1: float,
    s1: float,
    m2: float,
    s2: float,
    n1: int,
    epsilon: float,
    n2: Optional[int] = None,
    alpha: float = 0.05,
) -> TestResult:
    """
    Two One-Sided Tests (TOST) equivalence test from summary statistics.

    Declares equivalence when the (1 − 2α)·100 % CI ⊆ (−ε, +ε).

    H₀₁ : diff ≤ −ε   rejected when P(T ≥ t₁) < α,  t₁ = (diff + ε) / se
    H₀₂ : diff ≥ +ε   rejected when P(T ≤ t₂) < α,  t₂ = (diff − ε) / se
    TOST p-value = max(p₁, p₂)
    """
    n2 = n2 or n1
    diff, df, p_welch, se = welch_ttest(m1, s1, m2, s2, n1, n2)

    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lo = diff - t_crit * se
    ci_hi = diff + t_crit * se

    p_tost = max(
        stats.t.sf((diff + epsilon) / se, df),  # p₁  (upper one-sided)
        stats.t.cdf((diff - epsilon) / se, df),  # p₂  (lower one-sided)
    )
    equivalent = (ci_lo > -epsilon) and (ci_hi < epsilon)

    return TestResult(diff, ci_lo, ci_hi, p_welch, p_tost, equivalent)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "~"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# Console Reporting
# ══════════════════════════════════════════════════════════════════════════════


def print_analysis(
    rows: list[tuple[int, float, float, float, float]],
    n: int,
    epsilons: list[float],
    label: str,
    col1: str = "Method 1",
    col2: str = "Method 2",
) -> None:
    """
    Print TOST + Welch analysis for a list of (size, m1, s1, m2, s2) rows.
    """
    print(f"\n{'#' * 78}")
    print(f"# {label}")
    print(f"# diff = {col1} − {col2}")
    print(f"{'#' * 78}")

    for eps in epsilons:
        print(f"\n  ε = {eps}  →  margin [−{eps}, +{eps}]")
        print(
            f"  {'|P|':>5}  {'diff':>8}  {'90 % CI':>22}  "
            f"{'TOST p':>9}  {'≡':>3}  {'Welch p':>9}  {'sig':>4}"
        )
        print(f"  {'-' * 70}")
        for size, m1, s1, m2, s2 in rows:
            r = run_tost(m1, s1, m2, s2, n, eps)
            flag = "✅" if r.equivalent else "❌"
            print(
                f"  {size:>5}  {r.diff:>8.3f}  "
                f"[{r.ci_lo:>8.3f}, {r.ci_hi:>7.3f}]  "
                f"{r.p_tost:>9.4f}  {flag}  "
                f"{r.p_welch:>9.4f}  {r.stars:>4}"
            )

    print(
        "\n  *** p<.001  ** p<.01  * p<.05  ~ p<.1  ns = not significant"
        "\n  ≡  TOST equivalence (90 % CI ⊆ [−ε, +ε] and TOST p < 0.05)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX Table Generation
# ══════════════════════════════════════════════════════════════════════════════


def _cell(mean: float, std: float) -> str:
    """Format mean ± std for a LaTeX table cell."""
    return f"${mean:.2f}_{{\\pm {std:.2f}}}$"


def _pval(p: float) -> str:
    """
    Compact p-value for a table cell.
    <0.001 shown as <.001; otherwise 3 decimal places, leading zero stripped.
    Bolded when significant (p < 0.05).
    """
    if p < 0.001:
        inner = r"<\!.001"
    else:
        inner = f"{p:.3f}".lstrip("0")  # 0.046 → .046

    return f"$\\mathbf{{{inner}}}$" if p < 0.05 else f"${inner}$"


def _equiv_marker(equivalent: bool) -> str:
    """Small marker appended to TOST p-value cell when equivalence holds."""
    return r"$^{\approx}$" if equivalent else ""


def generate_latex_table(
    data: dict[int, tuple[MethodStats, MethodStats, MethodStats]],
    n: int,
    eps_vs_hs: float,
    eps_vs_nj: float,
    caption: str,
    label: str,
    placement: str = "hb!",
    resize: str = r"\textwidth",
) -> str:
    """
    Build a LaTeX ``table*`` with columns for NJ / HS / RHS and four
    significance sub-columns (RHS vs HS and RHS vs NJ, each showing
    the Welch p-value and the TOST p-value).

    Column layout (11 total):
        |P| | NJ RED | NJ CPU | HS RED | HS CPU | RHS RED | RHS CPU
             | Welch p vs HS | TOST p vs HS
             | Welch p vs NJ | TOST p vs NJ

    Formatting conventions
    ----------------------
    * p-values < 0.05 are **bolded**; < 0.001 shown as <.001
    * TOST p-value cell gets a small ≈ superscript when equivalence
      is declared (90 % CI ⊆ (−ε, +ε)  and  TOST p < 0.05)
    """
    L: list[str] = []
    ap = L.append

    # ── preamble ──────────────────────────────────────────────────────────────
    ap(f"\\begin{{table*}}[{placement}]")
    ap(r"\centering")
    ap(f"\\resizebox{{{resize}}}{{!}}{{")
    # 1 label + 2+2+2 metric cols + 2+2 sig cols = 11
    ap(r"\begin{tabular}{lcccccc cccc}")
    ap(r"\toprule")

    # ── header row 1 ──────────────────────────────────────────────────────────
    ap(
        r"\multicolumn{1}{c}{}"
        r" & \multicolumn{2}{c}{NJ}"
        r" & \multicolumn{2}{c}{HS}"
        r" & \multicolumn{2}{c}{RHS (ours)}"
        r" & \multicolumn{2}{c}{vs HS (RED $\uparrow$)}"
        r" & \multicolumn{2}{c}{vs NJ (RED $\uparrow$)}"
        r" \\"
    )
    ap(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        r" \cmidrule(lr){8-9} \cmidrule(lr){10-11}"
    )

    # ── header row 2 ──────────────────────────────────────────────────────────
    # p_Welch ↓ : small p = RHS significantly differs from (beats) baseline
    # p_TOST  ↓ : small p = strong evidence of practical equivalence
    ap(
        r"$|P|$"
        r" & RED\,$(\uparrow)$ & CPU"
        r" & RED\,$(\uparrow)$ & CPU"
        r" & RED\,$(\uparrow)$ & CPU"
        r" & $p_{\text{Welch}}$ & $p_{\text{TOST}}$"
        r" & $p_{\text{Welch}}$ & $p_{\text{TOST}}$"
        r" \\"
    )
    ap(r"\midrule")

    # ── data rows ─────────────────────────────────────────────────────────────
    for size in sorted(data):
        nj, hs, rhs = data[size]

        r_hs = run_tost(
            rhs.red_mean, rhs.red_std, hs.red_mean, hs.red_std, n, eps_vs_hs
        )
        r_nj = run_tost(
            rhs.red_mean, rhs.red_std, nj.red_mean, nj.red_std, n, eps_vs_nj
        )

        ap(
            f"    {size}"
            f" & {_cell(nj.red_mean,  nj.red_std)}"
            f" & {_cell(nj.cpu_mean,  nj.cpu_std)}"
            f" & {_cell(hs.red_mean,  hs.red_std)}"
            f" & {_cell(hs.cpu_mean,  hs.cpu_std)}"
            f" & {_cell(rhs.red_mean, rhs.red_std)}"
            f" & {_cell(rhs.cpu_mean, rhs.cpu_std)}"
            f" & {_pval(r_hs.p_welch)}"
            f" & {_pval(r_hs.p_tost)}{_equiv_marker(r_hs.equivalent)}"
            f" & {_pval(r_nj.p_welch)}"
            f" & {_pval(r_nj.p_tost)}{_equiv_marker(r_nj.equivalent)}"
            r" \\"
        )

    # ── footer ────────────────────────────────────────────────────────────────
    ap(r"\bottomrule")
    ap(r"\end{tabular}")
    ap(r"}")  # closes \resizebox
    ap(f"\\caption{{{caption}}}")
    ap(f"\\label{{{label}}}")
    ap(r"\end{table*}")

    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
# Data  —  edit this section for your own experiments
#
# Format: {|P|: (NJ, HS, RHS)}
# Each tuple element is MS(red_mean, red_std, cpu_mean, cpu_std).
# ══════════════════════════════════════════════════════════════════════════════

N_RUNS = 10  # number of independent runs used to compute each mean / std

# ── Table 1: Centered Gaussian G(0, 0.5) ─────────────────────────────────────

T1: dict[int, tuple[MethodStats, MethodStats, MethodStats]] = {
    10: (
        MS(1.46, 1.12, 5.50, 0.03),
        MS(2.72, 0.97, 0.03, 0.01),
        MS(3.17, 1.59, 4.60, 3.40),
    ),
    20: (
        MS(-2.60, 3.00, 5.68, 0.04),
        MS(1.83, 0.70, 0.06, 0.01),
        MS(3.15, 0.56, 9.87, 7.00),
    ),
    30: (
        MS(-1.70, 2.31, 5.86, 0.03),
        MS(2.12, 0.60, 0.10, 0.02),
        MS(3.03, 1.09, 12.65, 6.94),
    ),
    40: (
        MS(-2.62, 1.35, 5.95, 0.04),
        MS(2.17, 0.71, 0.13, 0.02),
        MS(3.30, 0.75, 17.59, 6.97),
    ),
    50: (
        MS(-4.37, 1.87, 6.14, 0.04),
        MS(2.26, 0.43, 0.17, 0.02),
        MS(3.08, 0.34, 26.21, 8.67),
    ),
    60: (
        MS(-3.38, 1.09, 6.29, 0.03),
        MS(2.56, 0.57, 0.20, 0.02),
        MS(2.72, 0.48, 26.95, 8.40),
    ),
    70: (
        MS(-3.28, 1.48, 6.51, 0.04),
        MS(2.45, 0.54, 0.22, 0.01),
        MS(2.68, 0.66, 31.09, 11.84),
    ),
    80: (
        MS(-3.89, 1.79, 6.77, 0.03),
        MS(2.43, 0.53, 0.26, 0.02),
        MS(3.07, 0.39, 38.55, 8.45),
    ),
    90: (
        MS(-3.46, 1.65, 7.01, 0.04),
        MS(2.72, 0.50, 0.30, 0.02),
        MS(2.87, 0.57, 66.07, 30.15),
    ),
    100: (
        MS(-3.77, 1.75, 7.24, 0.03),
        MS(2.45, 0.35, 0.33, 0.02),
        MS(2.84, 0.47, 70.31, 14.92),
    ),
    150: (
        MS(-4.17, 1.10, 8.46, 0.03),
        MS(2.27, 0.20, 0.48, 0.02),
        MS(2.48, 0.22, 139.87, 49.03),
    ),
    200: (
        MS(-5.01, 1.40, 11.09, 0.06),
        MS(2.43, 0.26, 0.66, 0.04),
        MS(2.64, 0.23, 341.42, 96.02),
    ),
}

# ── Table 2: Mixture of 10 Gaussians ─────────────────────────────────────────

T2: dict[int, tuple[MethodStats, MethodStats, MethodStats]] = {
    10: (
        MS(40.49, 0.03, 5.52, 0.04),
        MS(17.75, 4.57, 0.02, 0.01),
        MS(28.33, 18.54, 25.70, 13.30),
    ),
    20: (
        MS(40.10, 0.06, 5.73, 0.05),
        MS(14.35, 4.61, 0.05, 0.02),
        MS(40.01, 0.09, 61.94, 16.77),
    ),
    30: (
        MS(39.93, 0.08, 5.91, 0.04),
        MS(14.74, 5.75, 0.08, 0.06),
        MS(39.84, 0.13, 95.05, 27.50),
    ),
    40: (
        MS(39.70, 0.09, 6.06, 0.03),
        MS(16.63, 4.46, 0.09, 0.03),
        MS(39.71, 0.12, 119.86, 53.15),
    ),
    50: (
        MS(39.53, 0.06, 6.00, 0.02),
        MS(17.88, 5.29, 0.10, 0.04),
        MS(39.52, 0.05, 116.38, 45.92),
    ),
    60: (
        MS(39.22, 0.10, 6.37, 0.04),
        MS(14.95, 4.60, 0.11, 0.02),
        MS(39.36, 0.11, 87.99, 9.33),
    ),
    70: (
        MS(39.01, 0.16, 6.55, 0.05),
        MS(14.42, 4.27, 0.11, 0.03),
        MS(39.26, 0.05, 118.26, 24.23),
    ),
    80: (
        MS(38.55, 0.11, 6.77, 0.04),
        MS(14.47, 4.29, 0.12, 0.03),
        MS(39.11, 0.14, 111.56, 12.72),
    ),
    90: (
        MS(38.18, 0.18, 6.96, 0.06),
        MS(15.85, 3.70, 0.14, 0.03),
        MS(39.03, 0.08, 171.84, 59.68),
    ),
    100: (
        MS(37.50, 0.28, 7.26, 0.07),
        MS(13.91, 4.99, 0.13, 0.04),
        MS(38.94, 0.12, 183.57, 60.30),
    ),
    150: (
        MS(24.02, 1.24, 8.48, 0.05),
        MS(13.22, 6.21, 0.16, 0.02),
        MS(38.47, 0.10, 205.34, 52.63),
    ),
    200: (
        MS(-8.08, 1.09, 10.51, 0.06),
        MS(13.62, 4.43, 0.18, 0.03),
        MS(38.17, 0.17, 265.46, 80.59),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build row lists for print_analysis
# ══════════════════════════════════════════════════════════════════════════════


def _rows(
    table: dict[int, tuple[MethodStats, MethodStats, MethodStats]],
    m1_idx: int,
    m2_idx: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Extract (size, m1_red_mean, m1_red_std, m2_red_mean, m2_red_std).
    Index mapping: 0 = NJ, 1 = HS, 2 = RHS.
    """
    return [
        (
            size,
            v[m1_idx].red_mean,
            v[m1_idx].red_std,
            v[m2_idx].red_mean,
            v[m2_idx].red_std,
        )
        for size, v in sorted(table.items())
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Console analysis ──────────────────────────────────────────────────────
    print_analysis(
        _rows(T1, 2, 1),
        N_RUNS,
        [0.5, 1.0, 1.5],
        label="Table 1 — G(0, 0.5)  |  RHS vs HS",
        col1="RHS",
        col2="HS",
    )
    print_analysis(
        _rows(T1, 2, 0),
        N_RUNS,
        [1.0, 2.0],
        label="Table 1 — G(0, 0.5)  |  RHS vs NJ",
        col1="RHS",
        col2="NJ",
    )
    print_analysis(
        _rows(T2, 2, 1),
        N_RUNS,
        [5.0, 10.0, 15.0],
        label="Table 2 — Mixture    |  RHS vs HS",
        col1="RHS",
        col2="HS",
    )
    print_analysis(
        _rows(T2, 2, 0),
        N_RUNS,
        [0.5, 1.0, 2.0],
        label="Table 2 — Mixture    |  RHS vs NJ",
        col1="RHS",
        col2="NJ",
    )

    # ── LaTeX output ──────────────────────────────────────────────────────────
    SEP = "\n" + "=" * 78 + "\n"

    # Significance legend appended to every caption
    _sig_note = (
        r" Significance columns report Welch $t$-test and TOST p-values"
        r" (RHS vs.\ the indicated baseline, RED metric, $n{=}10$,"
        r" $\varepsilon{=}1.0$)."
        r" \textbf{Bold} indicates $p{<}0.05$;"
        r" ${}^{\approx}$ marks TOST equivalence"
        r" (90\,\% CI $\subseteq (-\varepsilon,+\varepsilon)$ and $p_{\text{TOST}}{<}0.05$)."
        r" Interpretation:"
        r" \begin{tabular}[t]{@{}ll@{}}"
        r" \toprule"
        r" $p_{\text{Welch}}$ sig.,\ $p_{\text{TOST}}$ not & Methods are detectably different.\\"
        r" $p_{\text{Welch}}$ not sig.,\ $p_{\text{TOST}}$ not & Inconclusive (underpowered).\\"
        r" $p_{\text{Welch}}$ not sig.,\ $p_{\text{TOST}}$ sig.~(${}^{\approx}$) & Methods are statistically equivalent.\\"
        r" Both significant & Difference exists but is practically negligible.\\"
        r" \bottomrule"
        r" \end{tabular}"
    )

    print(SEP + "LaTeX — Table 1" + SEP)
    print(
        generate_latex_table(
            T1,
            N_RUNS,
            eps_vs_hs=1.0,
            eps_vs_nj=1.0,
            caption=(
                r"Sampling $|P|$ points from a centered hyperbolic Gaussian"
                r" $\mathcal{G}(0, 0.5)$."
                r" RED: reduction over MST (\%). CPU: total CPU time (sec.)."
                + _sig_note
            ),
            label="tab:exp1",
        )
    )

    print(SEP + "LaTeX — Table 2" + SEP)
    print(
        generate_latex_table(
            T2,
            N_RUNS,
            eps_vs_hs=1.0,
            eps_vs_nj=1.0,
            caption=(
                r"Sampling $|P|$ points from a mixture of $10$ hyperbolic Gaussians"
                r" $\mathcal{G}(\mu_{10,k}(1-10^{-10}), 0.1)$,"
                r" $k \in \{1, \ldots, 10\}$."
                r" RED: reduction over MST (\%). CPU: total CPU time (sec.)."
                + _sig_note
            ),
            label="tab:exp2",
        )
    )
