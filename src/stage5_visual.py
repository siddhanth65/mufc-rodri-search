"""
stage5_visual.py — Visual layer: percentile radar grid + comparison table.

Produces two artifacts for the README / repo:
  1. data/processed/shortlist_radar.html  — interactive (hover for values)
  2. data/processed/shortlist_radar.png   — static, for embedding in README
  3. data/processed/shortlist_table.png   — DataMB-style percentile table

Reads stage4_profiles.parquet (already has pct_ columns + scores).
Small-multiples radar: one mini-radar per player, shared axes, grid layout.
This is readable where a single overlaid 12-player radar is not.

Usage:
    python src/stage5_visual.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/processed")
STAGE4_PATH = OUT_DIR / "stage4_profiles.parquet"
N_PLAYERS = 12  # top N by stage3_rank (non-seed)

# The 8 axes for the radar — (pct_column, short_label)
RADAR_AXES = [
    ("pct_recoveries", "Recoveries"),
    ("pct_tackles_won_pct", "Tackle\nwin %"),
    ("pct_interceptions", "Interceptions"),
    ("pct_own_half_passes", "Own-half\npasses"),
    ("pct_pass_cmp_pct", "Pass\ncomp %"),
    ("pct_passes_final_third", "Final-third\npasses"),
    ("pct_possession_lost", "Ball\nsecurity"),
    ("pct_key_passes", "Key\npasses"),
]

# Table columns — (pct_column, header)
TABLE_COLS = [
    ("pct_recoveries", "Recov"),
    ("pct_tackles_won_pct", "Tkl%"),
    ("pct_interceptions", "Int"),
    ("pct_own_half_passes", "OwnHlf"),
    ("pct_pass_cmp_pct", "Pass%"),
    ("pct_passes_final_third", "F3 Pass"),
    ("pct_possession_lost", "BallSec"),
    ("pct_key_passes", "KeyP"),
]


# Colour ramp for percentile cells (low -> high)
def pct_color(v):
    if pd.isna(v):
        return "#e8e8e8"
    v = float(v)
    # red (low) -> amber (mid) -> green (high)
    if v < 50:
        # red to amber
        t = v / 50
        r, g, b = 0.83, 0.20 + 0.55 * t, 0.20
    else:
        # amber to green
        t = (v - 50) / 50
        r, g, b = 0.83 - 0.55 * t, 0.75, 0.20 + 0.15 * t
    return (r, g, b)


# ---------------------------------------------------------------------------
# Radar — small multiples
# ---------------------------------------------------------------------------


def draw_radar_grid(df: pd.DataFrame, axes_spec, out_png: Path) -> None:
    n = len(df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    labels = [lab for _, lab in axes_spec]
    cols = [c for c, _ in axes_spec]
    n_ax = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, axarr = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.1, nrows * 3.4),
        subplot_kw=dict(polar=True),
    )
    axarr = np.array(axarr).reshape(-1)

    fig.suptitle(
        "Defensive-Midfield Shortlist — Percentile Profiles",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "Each axis: percentile rank vs all main-track midfielders (top-7 leagues, 3 seasons). Outer edge = 100th.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )

    accent = "#c1121f"

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axarr[i]
        vals = [row.get(c, np.nan) for c in cols]
        vals = [0 if pd.isna(v) else v for v in vals]
        vals += vals[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["25", "50", "75"], fontsize=6, color="#aaa")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.tick_params(pad=1)

        ax.plot(angles, vals, color=accent, linewidth=1.6)
        ax.fill(angles, vals, color=accent, alpha=0.25)

        rank = int(row["stage3_rank"]) if pd.notna(row.get("stage3_rank")) else "?"
        age = f"{int(row['tm_age'])}" if pd.notna(row.get("tm_age")) else "?"
        title = f"#{rank}  {row['player']}"
        sub = f"{row.get('team','')} · age {age}"
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=14)
        ax.text(
            0.5,
            -0.16,
            sub,
            transform=ax.transAxes,
            ha="center",
            fontsize=6.8,
            color="#666",
        )

    # hide unused cells
    for j in range(n, len(axarr)):
        axarr[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  radar grid -> {out_png}")


# ---------------------------------------------------------------------------
# Percentile table (DataMB-style)
# ---------------------------------------------------------------------------


def draw_percentile_table(df: pd.DataFrame, table_cols, out_png: Path) -> None:
    headers = ["#", "Player", "Team", "Age"] + [h for _, h in table_cols]
    n_rows = len(df)
    n_cols = len(headers)

    row_h = 0.42
    fig_h = 1.2 + n_rows * row_h
    fig_w = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1.5)

    ax.text(
        0,
        n_rows + 1.0,
        "Defensive-Midfield Shortlist — Percentile Table",
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0,
        n_rows + 0.55,
        "0-100 percentile vs all main-track midfielders. Green = elite, red = weak for the role.",
        fontsize=8.5,
        color="#666",
        va="bottom",
    )

    # column x-positions: wide name/team cols, narrow stat cols
    col_x = [0.0, 0.5, 2.7, 4.5]
    stat_start = 5.2
    stat_w = (n_cols - 5.2) / len(table_cols)
    for k in range(len(table_cols)):
        col_x.append(stat_start + k * stat_w)
    col_x.append(n_cols)

    # header row
    y_head = n_rows + 0.15
    for c, h in enumerate(headers):
        ax.text(
            col_x[c] + 0.05,
            y_head,
            h,
            fontsize=8.5,
            fontweight="bold",
            va="center",
            color="#222",
        )
    ax.plot([0, n_cols], [y_head - 0.25, y_head - 0.25], color="#333", lw=1.0)

    for i, (_, row) in enumerate(df.iterrows()):
        y = n_rows - 1 - i + 0.5
        rank = int(row["stage3_rank"]) if pd.notna(row.get("stage3_rank")) else "?"
        age = f"{int(row['tm_age'])}" if pd.notna(row.get("tm_age")) else "?"

        ax.text(col_x[0] + 0.05, y, str(rank), fontsize=8, va="center", color="#444")
        ax.text(
            col_x[1] + 0.05,
            y,
            str(row["player"])[:22],
            fontsize=8,
            va="center",
            fontweight="bold",
        )
        ax.text(
            col_x[2] + 0.05,
            y,
            str(row.get("team", ""))[:20],
            fontsize=7.5,
            va="center",
            color="#666",
        )
        ax.text(col_x[3] + 0.05, y, age, fontsize=8, va="center", color="#444")

        for k, (pcol, _) in enumerate(table_cols):
            v = row.get(pcol, np.nan)
            cx = col_x[4 + k]
            cw = stat_w
            ax.add_patch(
                plt.Rectangle(
                    (cx, y - 0.19),
                    cw - 0.08,
                    0.38,
                    facecolor=pct_color(v),
                    edgecolor="white",
                    lw=0.5,
                )
            )
            txt = "—" if pd.isna(v) else f"{int(round(v))}"
            ax.text(
                cx + (cw - 0.08) / 2,
                y,
                txt,
                fontsize=7.5,
                va="center",
                ha="center",
                color="#1a1a1a",
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  percentile table -> {out_png}")


# ---------------------------------------------------------------------------
# Interactive HTML (plotly) — overlaid radar with toggle
# ---------------------------------------------------------------------------


def draw_interactive(df: pd.DataFrame, axes_spec, out_html: Path) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed — skipping interactive (pip install plotly)")
        return

    labels = [lab.replace("\n", " ") for _, lab in axes_spec]
    cols = [c for c, _ in axes_spec]

    fig = go.Figure()
    for _, row in df.iterrows():
        vals = [row.get(c, 0) or 0 for c in cols]
        vals += vals[:1]
        rank = int(row["stage3_rank"]) if pd.notna(row.get("stage3_rank")) else "?"
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=labels + labels[:1],
                name=f"#{rank} {row['player']}",
                fill="toself",
                opacity=0.55,
                visible="legendonly" if rank not in (5, 6, 7) else True,
            )
        )

    fig.update_layout(
        title="Defensive-Midfield Shortlist — Percentile Profiles "
        "(click legend to toggle players)",
        polar=dict(radialaxis=dict(range=[0, 100], showline=True)),
        showlegend=True,
        width=900,
        height=700,
    )
    fig.write_html(str(out_html))
    print(f"  interactive radar -> {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = pd.read_parquet(STAGE4_PATH)

    # top N non-seed by stage3_rank; stage4 file already excludes seeds in profiles
    df = df.sort_values("stage3_rank").head(N_PLAYERS).reset_index(drop=True)
    print(f"Visualising top {len(df)} shortlist players\n")

    draw_radar_grid(df, RADAR_AXES, OUT_DIR / "shortlist_radar.png")
    draw_percentile_table(df, TABLE_COLS, OUT_DIR / "shortlist_table.png")
    draw_interactive(df, RADAR_AXES, OUT_DIR / "shortlist_radar.html")

    print("\nDone. Embed shortlist_radar.png and shortlist_table.png in README;")
    print("commit shortlist_radar.html as the interactive version.")


if __name__ == "__main__":
    main()
