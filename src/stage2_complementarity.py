"""
stage2_complementarity.py — Stage 2: Complementarity ranking.

Takes the top 80 from Stage 1 (ranked_main.parquet) and re-ranks by fit to
United's specific midfield need: completing a trio with Kobbie Mainoo and
Bruno Fernandes.

Steps:
  1. Rebuild fingerprints (same ref_pool as Stage 1)
  2. Compute Mainoo + Bruno's average z-vector ("existing midfield")
  3. Gap = population_mean − existing_midfield (features they lack)
  4. Complementarity score: weighted dot product of candidate vs positive gap
  5. Combined score: 0.5 × archetype_score + 0.5 × complementarity_score
  6. League tier modifier (soft, not a hard filter)
  7. Age flag from raw Sofascore data

Usage:
    python src/stage2_complementarity.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sofascore_pipeline handles the UTF-8 stdout fix at import time — don't double-wrap
sys.path.insert(0, str(Path(__file__).parent))
from sofascore_pipeline import (
    FEATURES,
    WEIGHTS,
    LEAGUES,
    SEASONS,
    CACHE_DIR,
    MIN_MINUTES_MAIN,
    MIN_MINUTES_PROSPECT,
    load_and_aggregate,
    build_fingerprint,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/processed")
STAGE1_PATH = OUT_DIR / "ranked_main.parquet"

TOP_N = 80

ALPHA_ARCHETYPE = 0.5
ALPHA_COMPLEMENTARITY = 0.5

LEAGUE_TIER = {
    "England Premier League": 1.00,
    "Spain La Liga":          1.00,
    "Germany Bundesliga":     1.00,
    "Italy Serie A":          1.00,
    "France Ligue 1":         1.00,
    "Netherlands Eredivisie": 0.85,
    "Portugal Primeira Liga": 0.85,
}

EXISTING_PAIR = ["Kobbie Mainoo", "Bruno Fernandes"]

# Players to explicitly check in the scout interest summary
SCOUT_WATCH = [
    "Carlos Baleba",
    "Matheus Fernandes",
    "Boubacar Kamara",
    "Lamine Camara",
    "Ryan Gravenberch",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_player_ages() -> pd.DataFrame:
    """Pull most-recent-season age for every player from the raw cached files.
    Sofascore API currently does not return an 'age' field — returns empty frame;
    age_flag will be populated in Stage 3 via Transfermarkt.
    """
    frames = []
    season_order = {"25/26": 3, "24/25": 2, "23/24": 1}
    for league in LEAGUES:
        for season in SEASONS:
            safe = f"{league.replace(' ', '_')}__{season.replace('/', '_')}.parquet"
            cache = CACHE_DIR / safe
            if not cache.exists():
                continue
            df = pd.read_parquet(cache)
            if "age" not in df.columns:
                continue
            sub = df[["player", "age"]].copy()
            sub["_srank"] = season_order.get(season, 0)
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["player", "age"])  # no age data available yet
    all_ages = pd.concat(frames, ignore_index=True)
    all_ages["age"] = pd.to_numeric(all_ages["age"], errors="coerce")
    latest = (
        all_ages.dropna(subset=["age"])
        .sort_values("_srank", ascending=False)
        .drop_duplicates("player")[["player", "age"]]
    )
    return latest


def find_player(fp: pd.DataFrame, name: str) -> pd.Series | None:
    last = name.split()[-1]
    hits = fp[fp["player"].str.contains(last, case=False, na=False)]
    if hits.empty:
        return None
    return hits.sort_values("minutes", ascending=False).iloc[0]


def age_flag(age) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if a < 24:
        return "young"
    if a <= 29:
        return "prime"
    return "veteran"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Step 0: Load Stage 1 rankings ───────────────────────────────────────
    ranked = pd.read_parquet(STAGE1_PATH)
    top80_names = set(
        ranked[ranked["archetype_rank"] <= TOP_N]["player"].tolist()
    )
    print(f"Stage 1 pool: {len(top80_names)} players (top {TOP_N})")

    # ── Step 1: Rebuild fingerprint (same logic as Stage 1) ─────────────────
    agg = load_and_aggregate()
    main_pool = agg[agg["minutes"] >= MIN_MINUTES_MAIN].copy().reset_index(drop=True)

    # ref_pool for z-scoring: always the main-track pool, same as Stage 1
    ref_copy = main_pool.copy()
    main_fp  = build_fingerprint(main_pool.copy(), ref_pool=ref_copy)

    # Also fingerprint sub-threshold players so we can locate Mainoo/Bruno if needed
    sub_pool = agg[
        (agg["minutes"] >= MIN_MINUTES_PROSPECT)
        & (agg["minutes"] < MIN_MINUTES_MAIN)
    ].copy().reset_index(drop=True)
    ref_copy2 = main_pool.copy()  # fresh copy; ref_pool must be unfingerprinted
    sub_fp = build_fingerprint(sub_pool.copy(), ref_pool=ref_copy2) if not sub_pool.empty else pd.DataFrame()

    all_fp = pd.concat([main_fp, sub_fp], ignore_index=True) if not sub_fp.empty else main_fp

    feat_names = [f[0] for f in FEATURES if f"z_{f[0]}" in main_fp.columns]
    z_cols     = [f"z_{f}" for f in feat_names]
    weights    = np.array([WEIGHTS.get(f, 1.0) for f in feat_names])

    # ── Step 2: Existing midfield vector ────────────────────────────────────
    print()
    pair_vecs = []
    for name in EXISTING_PAIR:
        row = find_player(all_fp, name)
        if row is None:
            raise RuntimeError(f"Could not find {name!r} in fingerprint")
        pair_vecs.append(row[z_cols].fillna(0).to_numpy(dtype=float))
        print(f"Found: {row['player']:30s} ({row['team']}, {int(row['minutes'])} mins)")

    existing_z = np.mean(pair_vecs, axis=0)  # (n_features,)

    # ── Step 3: Gap dimensions ───────────────────────────────────────────────
    # gap[i] > 0 => duo below population mean => #6 must cover this dimension
    gap          = 0.0 - existing_z
    positive_gap = np.maximum(gap, 0.0)

    print("\n" + "=" * 64)
    print("GAP DIMENSIONS — what Mainoo + Bruno lack, #6 must cover")
    print("=" * 64)
    gap_order = np.argsort(gap)[::-1]
    print(f"\n  {'feature':25s} {'mainoo_z':>10s} {'bruno_z':>10s} {'pair_avg':>10s} {'gap':>8s}")
    for i in gap_order[:8]:
        print(
            f"  {feat_names[i]:25s} {pair_vecs[0][i]:>10.2f}"
            f" {pair_vecs[1][i]:>10.2f} {existing_z[i]:>10.2f} {gap[i]:>8.2f}"
        )

    # ── Step 4: Complementarity score ───────────────────────────────────────
    # Pull archetype columns from Stage 1 into the main fingerprint
    main_fp = main_fp.merge(
        ranked[["player", "archetype_rank", "archetype_distance"]],
        on="player", how="left"
    )

    candidates = main_fp[main_fp["player"].isin(top80_names)].copy()
    n = len(candidates)
    print(f"\nCandidates with fingerprint data: {n}")

    Z        = candidates[z_cols].fillna(0).values             # (n, n_feats)
    gap_w    = positive_gap * weights                           # (n_feats,)
    comp_raw = (Z * gap_w[None, :]).sum(axis=1)                # (n,)

    candidates["complementarity_raw"] = comp_raw
    cmin, cmax = comp_raw.min(), comp_raw.max()
    candidates["complementarity_score"] = (
        (comp_raw - cmin) / (cmax - cmin) if cmax > cmin else np.full(n, 0.5)
    )

    # ── Step 5: Normalize archetype distance to [0, 1] ──────────────────────
    dist = candidates["archetype_distance"].fillna(candidates["archetype_distance"].max()).values
    dmin, dmax = dist.min(), dist.max()
    arch_score = (
        1.0 - (dist - dmin) / (dmax - dmin) if dmax > dmin else np.ones(n)
    )
    candidates["archetype_score"] = arch_score

    # ── Step 6: Combined score + league tier modifier ────────────────────────
    candidates["combined_raw"] = (
        ALPHA_ARCHETYPE       * candidates["archetype_score"]
        + ALPHA_COMPLEMENTARITY * candidates["complementarity_score"]
    )
    candidates["league_tier"]    = candidates["league"].map(LEAGUE_TIER).fillna(1.0)
    candidates["combined_score"] = candidates["combined_raw"] * candidates["league_tier"]

    # ── Step 7: Age flag ─────────────────────────────────────────────────────
    ages = get_player_ages()
    candidates = candidates.merge(ages, on="player", how="left")
    candidates["age_flag"] = candidates["age"].apply(age_flag)

    # ── Output ───────────────────────────────────────────────────────────────
    candidates = (
        candidates.sort_values("combined_score", ascending=False)
        .reset_index(drop=True)
    )
    candidates["stage2_rank"] = candidates.index + 1

    display_cols = [
        "player", "team", "league", "minutes", "age", "age_flag",
        "archetype_rank", "complementarity_score", "combined_score", "league_tier",
    ]

    print("\n" + "=" * 64)
    print(f"STAGE 2 — TOP 30  (archetype + complementarity, league-adjusted)")
    print("=" * 64)
    print(
        candidates[display_cols]
        .head(30)
        .to_string(index=False, float_format="{:.3f}".format)
    )

    # Save
    save_cols = display_cols + ["stage2_rank", "archetype_score", "combined_raw",
                                "complementarity_raw"]
    candidates[save_cols].to_parquet(OUT_DIR / "stage2_ranked.parquet", index=False)
    print(f"\nSaved: {OUT_DIR / 'stage2_ranked.parquet'}")

    # ── Scout interest check ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SCOUT WATCH — Stage 2 positions")
    print("=" * 64)
    for name in SCOUT_WATCH:
        last = name.split()[-1]
        hit = candidates[candidates["player"].str.contains(last, case=False, na=False)]
        if hit.empty:
            print(f"  {'—':>4s}  {name:30s}  not in top {TOP_N} (fell out at Stage 1)")
        else:
            r = hit.iloc[0]
            age_str = f"{int(r['age'])}" if pd.notna(r.get("age")) else "?"
            print(
                f"  #{int(r['stage2_rank']):3d}  {r['player']:30s}"
                f"  arch={int(r['archetype_rank']):3d}"
                f"  comp={r['complementarity_score']:.3f}"
                f"  combined={r['combined_score']:.3f}"
                f"  age={age_str} ({r['age_flag']})"
            )


if __name__ == "__main__":
    main()
