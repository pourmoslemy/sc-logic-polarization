"""Generate Elsevier CAS-ready figures from authoritative Phase 1 results.

Outputs both 300-DPI PNG files and vector PDF companions.  The dimensions are
chosen for the single-column CAS submission layout while remaining suitable for
full-width ``figure*`` use in a future double-column conversion.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = PROJECT_ROOT / "generated_results"
OUTPUT = SCRIPT_DIR / "sc_logic_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {104: "#0072B2", 107: "#D55E00", 108: "#009E73"}
PARTY_COLORS = {"R": "#C44E52", "D": "#4C72B0"}


def configure_style() -> None:
    """Apply a restrained, colorblind-safe Elsevier-compatible visual style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[dict]:
    """Save raster and vector versions and return manifest entries."""
    outputs = []
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white", **options)
        outputs.append({"path": str(path.relative_to(PROJECT_ROOT)), "format": suffix})
    plt.close(fig)
    return outputs


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(RESULTS / "reproduced_cross_congress_ci.csv")
    audit = pd.read_csv(RESULTS / "reproduced_antitonicity_audit.csv")
    expected = {104, 107, 108}
    if set(metrics["congress"]) != expected or len(metrics) != 72:
        raise ValueError("Expected 24 stages for each of Congresses 104, 107, and 108")
    if len(audit) != 1656 or audit["violation_faces"].sum() != 0:
        raise ValueError("Antitonicity audit is incomplete or contains violations")
    return metrics, audit


def plot_cross_congress_ci(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Plot the primary cross-Congress CI_A trajectories."""
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    for congress in (104, 107, 108):
        frame = metrics[metrics.congress == congress]
        ax.plot(
            frame.stage,
            frame.CI_A,
            color=COLORS[congress],
            lw=2.2,
            marker="o",
            ms=3.2,
            markevery=[0, 5, 11, 17, 23],
            label=f"{congress}th Senate",
        )
        ax.scatter(frame.stage.iloc[-1], frame.CI_A.iloc[-1], s=45, color=COLORS[congress], zorder=4)
        label_offset = {104: (6, -11), 107: (6, 10), 108: (6, 1)}[congress]
        ax.annotate(
            f"{frame.CI_A.iloc[-1]:.3f}",
            (frame.stage.iloc[-1], frame.CI_A.iloc[-1]),
            xytext=label_offset,
            textcoords="offset points",
            va="center",
            color=COLORS[congress],
            fontsize=8,
        )
    ax.axvline(12.5, color="#777777", lw=0.8, ls="--")
    ax.text(12.65, 0.72, "Year boundary", color="#666666", fontsize=8, va="top")
    ax.set(xlabel="Monthly cumulative stage (1-24)", ylabel=r"Republican-core contraction index $CI_A(t)$")
    ax.set_xlim(1, 24.8)
    ax.set_ylim(0, max(0.75, metrics.CI_A.max() * 1.05))
    ax.legend(loc="upper right", frameon=True, ncol=1)
    ax.set_title("Cross-Congress persistence of graded simplicial negation")
    fig.tight_layout()
    caption = (
        "Monthly trajectories of the Republican-core contraction index for the 104th, 107th, "
        "and 108th Senates, reconstructed from raw Fowler SN/SP records on invariant classified "
        "actor cohorts. The index counts vertices, edges, and triangles in stage-indexed Heyting negation."
    )
    return save_figure(fig, "cross_congress_polarization_plot"), caption


def plot_sc_vs_modularity(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Contrast CI_A persistence with party-modularity compression in Congress 107."""
    frame = metrics[metrics.congress == 107].copy()
    fig, ax_ci = plt.subplots(figsize=(7.2, 4.4))
    ax_q = ax_ci.twinx()
    ax_ci.axvspan(9, 22, color="#F0E442", alpha=0.16, lw=0, label="Post-9/11 focal interval")
    line_ci = ax_ci.plot(frame.stage, frame.CI_A, color="#D55E00", lw=2.4, marker="o", ms=3.4, label=r"$CI_A(t)$")
    line_q = ax_q.plot(frame.stage, frame.Q_party_unweighted, color="#0072B2", lw=2.1, marker="s", ms=3.1, label=r"$Q(t)$")
    ax_q.axhline(0, color="#444444", lw=0.8, ls=":")
    ax_ci.set(xlabel="Monthly cumulative stage in the 107th Senate", ylabel=r"Contraction index $CI_A(t)$")
    ax_q.set_ylabel(r"Newman--Girvan party modularity $Q(t)$", color="#0072B2")
    ax_ci.tick_params(axis="y", colors="#D55E00")
    ax_q.tick_params(axis="y", colors="#0072B2")
    ax_ci.set_xlim(1, 24)
    ax_ci.set_ylim(0, max(0.75, frame.CI_A.max() * 1.04))
    q_pad = max(0.002, (frame.Q_party_unweighted.max() - frame.Q_party_unweighted.min()) * 0.12)
    ax_q.set_ylim(frame.Q_party_unweighted.min() - q_pad, frame.Q_party_unweighted.max() + q_pad)
    ax_ci.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
    labels = [frame.loc[frame.stage == s, "month"].iloc[0] for s in ax_ci.get_xticks()]
    ax_ci.set_xticklabels(labels, rotation=35, ha="right")
    handles = line_ci + line_q + [patches.Patch(facecolor="#F0E442", alpha=0.25, label="Post-9/11 focal interval")]
    ax_ci.legend(handles=handles, loc="upper right", frameon=True)
    ax_ci.set_title("Topological persistence versus scalar modularity in the 107th Senate")
    fig.tight_layout()
    caption = (
        "Dual-axis comparison for the 107th Senate. Party modularity rapidly approaches a weakly "
        "negative near-complete-graph regime, whereas proposition-relative simplicial negation retains "
        "positive face mass. The shaded region marks September 2001 through October 2002."
    )
    return save_figure(fig, "sc_vs_zhang_modularity_curve"), caption


def plot_phase_portrait(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Show non-equivalence of similar Q values and distinct CI_A states."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for congress in (104, 107, 108):
        frame = metrics[metrics.congress == congress]
        ax.plot(frame.Q_party_unweighted, frame.CI_A, color=COLORS[congress], alpha=0.85, lw=1.5)
        points = ax.scatter(
            frame.Q_party_unweighted,
            frame.CI_A,
            c=frame.stage,
            cmap="viridis",
            s=28,
            edgecolor=COLORS[congress],
            linewidth=0.6,
            label=f"{congress}th Senate",
        )
        ax.scatter(frame.Q_party_unweighted.iloc[0], frame.CI_A.iloc[0], marker="^", s=70, color=COLORS[congress], zorder=5)
        ax.scatter(frame.Q_party_unweighted.iloc[-1], frame.CI_A.iloc[-1], marker="X", s=70, color=COLORS[congress], zorder=5)
    cbar = fig.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label("Stage index")
    ax.axvline(0, color="#555555", lw=0.8, ls=":")
    ax.set(xlabel=r"Party modularity $Q(t)$", ylabel=r"Contraction index $CI_A(t)$")
    ax.set_title("Phase portrait: scalar similarity does not imply simplicial equivalence")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    caption = (
        "Joint trajectories in the (Q, CI_A) plane. Triangles mark stage 1 and crosses mark stage 24. "
        "The 107th and 108th Senates converge to nearly identical modularity while retaining distinct "
        "contraction indices, demonstrating the additional proposition-relative information in SC-Logic."
    )
    return save_figure(fig, "ci_modularity_phase_portrait"), caption


def plot_saturation(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Plot edge and triangle saturation of the cumulative 2-skeleta."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    for congress in (104, 107, 108):
        frame = metrics[metrics.congress == congress].copy()
        n = frame.actor_count.iloc[0]
        edge_max = n * (n - 1) / 2
        triangle_max = n * (n - 1) * (n - 2) / 6
        axes[0].plot(frame.stage, frame.f1 / edge_max, color=COLORS[congress], lw=2, label=f"{congress}th")
        axes[1].plot(frame.stage, frame.f2 / triangle_max, color=COLORS[congress], lw=2, label=f"{congress}th")
    axes[0].set(ylabel="Fraction of possible edges", xlabel="Stage", title="1-skeleton saturation")
    axes[1].set(ylabel="Fraction of possible triangles", xlabel="Stage", title="2-simplex saturation")
    for ax in axes:
        ax.set_xlim(1, 24)
        ax.set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Rapid combinatorial saturation under cumulative cosponsorship", y=1.02)
    fig.tight_layout()
    caption = (
        "Cumulative edge and triangle densities relative to the complete 2-skeleton on each invariant "
        "actor cohort. Rapid saturation explains why a graph scalar can lose temporal resolution while "
        "the proposition-indexed negative subcomplex remains discriminating."
    )
    return save_figure(fig, "simplicial_saturation_diagnostics"), caption


def plot_core_growth(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Plot emergence of Republican and Democratic core supports."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    for ax, congress in zip(axes, (104, 107, 108), strict=True):
        frame = metrics[metrics.congress == congress]
        ax.plot(frame.stage, frame.core_R_vertices / frame.republican_actors, color=PARTY_COLORS["R"], lw=2, label="Republican core")
        ax.plot(frame.stage, frame.core_D_vertices / frame.democratic_actors, color=PARTY_COLORS["D"], lw=2, label="Democratic core")
        ax.set_title(f"{congress}th Senate")
        ax.set_xlabel("Stage")
        ax.set_xlim(1, 24)
        ax.set_ylim(0, 1.04)
    axes[0].set_ylabel("Share of party actors in core")
    axes[-1].legend(loc="lower right", fontsize=7, frameon=True)
    fig.suptitle("Stagewise emergence of partisan proposition supports", y=1.03)
    fig.tight_layout()
    caption = (
        "Growth of party-core vertex supports under the three-qualifying-bill criterion. Core expansion "
        "changes the Heyting negation domain and therefore provides the order-theoretic mechanism behind "
        "contraction-index dynamics."
    )
    return save_figure(fig, "partisan_core_growth"), caption


def plot_audit(audit: pd.DataFrame) -> tuple[list[dict], str]:
    """Visualize the number of explicit faces checked for every strict stage pair."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharex=True, sharey=True)
    values = audit["intersection_faces_checked"].replace(0, np.nan)
    vmin, vmax = values.min(), values.max()
    image = None
    for ax, congress in zip(axes, (104, 107, 108), strict=True):
        subset = audit[(audit.congress == congress) & (audit.proposition == "p_R")]
        matrix = np.full((24, 24), np.nan)
        for row in subset.itertuples(index=False):
            matrix[row.s_stage - 1, row.t_stage - 1] = row.intersection_faces_checked
        image = ax.imshow(matrix, origin="lower", cmap="mako", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="equal")
        ax.set_title(f"{congress}th")
        ax.set_xlabel("Later stage t")
        ax.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
        ax.set_yticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
    axes[0].set_ylabel("Earlier stage s")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Faces in neg(t) intersect K_s")
    fig.suptitle("Explicit-face coverage of the Proposition 6.4 audit (all violations = 0)", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.17, top=0.82, wspace=0.22)
    caption = (
        "Pairwise audit coverage for Republican-core negation. Each populated cell represents explicit "
        "vertex, edge, and triangle identities tested for a strict pair s<t. The corresponding Democrat-core "
        "audit was also executed; all 1,656 proposition-pair rows across the three Congresses have zero violations."
    )
    return save_figure(fig, "antitonicity_audit_coverage"), caption


def plot_graphical_abstract(metrics: pd.DataFrame) -> tuple[list[dict], str]:
    """Create a compact graphical abstract compatible with the CAS front matter."""
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.25, 0.16, 0.5, "Raw Fowler data\nSN/SP records\n+ SH.sav", "#DCEAF7"),
        (0.22, 0.25, 0.16, 0.5, "Invariant D/R\nSenate cohort", "#E5F5E0"),
        (0.42, 0.25, 0.16, 0.5, "Monthly\ncumulative\n2-skeleton", "#FFF2CC"),
        (0.62, 0.25, 0.16, 0.5, "Heyting negation\n+ CI_A(t)", "#FADBD8"),
        (0.82, 0.25, 0.16, 0.5, "Q comparison\n+ 1,656-pair\naudit", "#E8DAEF"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#333333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, weight="semibold")
    for x in (0.18, 0.38, 0.58, 0.78):
        ax.annotate("", xy=(x + 0.035, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    terminal = metrics.sort_values(["congress", "stage"]).groupby("congress").tail(1)
    summary = "   ".join(f"{int(r.congress)}: CI_A={r.CI_A:.3f}, Q={r.Q_party_unweighted:.4f}" for r in terminal.itertuples())
    ax.text(0.5, 0.08, summary + "   |   Proposition 6.4: 0 violations", ha="center", va="center", fontsize=8.2)
    ax.text(0.5, 0.93, "SC-Logic raw-data reproduction: legislative events to order-theoretic invariants", ha="center", va="center", fontsize=11, weight="bold")
    fig.tight_layout()
    caption = (
        "Graphical abstract of the raw-data pipeline, showing the progression from Fowler legislative "
        "events to invariant cohorts, cumulative 2-skeleta, graded negation, scalar comparison, and exact antitonicity audit."
    )
    return save_figure(fig, "graphical_abstract"), caption


def main() -> None:
    configure_style()
    metrics, audit = load_results()
    builders = (
        ("cross_congress_polarization_plot", lambda: plot_cross_congress_ci(metrics)),
        ("sc_vs_zhang_modularity_curve", lambda: plot_sc_vs_modularity(metrics)),
        ("ci_modularity_phase_portrait", lambda: plot_phase_portrait(metrics)),
        ("simplicial_saturation_diagnostics", lambda: plot_saturation(metrics)),
        ("partisan_core_growth", lambda: plot_core_growth(metrics)),
        ("antitonicity_audit_coverage", lambda: plot_audit(audit)),
        ("graphical_abstract", lambda: plot_graphical_abstract(metrics)),
    )
    manifest = []
    captions = ["# Phase 2 figure captions", ""]
    for stem, builder in builders:
        files, caption = builder()
        manifest.append({"stem": stem, "files": files, "caption": caption})
        captions.extend([f"## {stem}", "", caption, ""])
        print(f"generated {stem}.png and {stem}.pdf")
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "FIGURE_CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    print(f"manifest: {OUTPUT / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
