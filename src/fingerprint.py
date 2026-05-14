"""
fingerprint.py — Build the 27-feature style fingerprint and run Stage 1.

Pipeline:
  1. Load all cached stat types
  2. Inspect columns across stat types (printed on first run)
  3. Filter to midfielders (primary_pos == MF, minutes >= 1500 across both seasons)
  4. Merge stat types per (player, team, league, season)
  5. Minutes-weighted aggregate across seasons -> one row per player
  6. Compute per-90 features
  7. PAdj defensive features (possession-adjusted)
  8. Z-score normalize within midfielder population
  9. Weighted Euclidean distance to seed vector (Rodri / Zubimendi / Tchouameni mean)
  10. Print ranked output and validate against pre-registered validation set

Usage:
    python src/fingerprint.py --inspect      # print columns available per stat type
    python src/fingerprint.py --run          # build fingerprint + run Stage 1
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Validation set (pre-registered in NOTES.md — do not change)
# ---------------------------------------------------------------------------

MUST_TOP_10 = ["Rodri", "Martin Zubimendi", "Aurélien Tchouaméni"]
MUST_TOP_50 = [
    "Declan Rice",
    "Moisés Caicedo",
    "Stanislav Lobotka",
    "Angelo Stiller",
    "Edson Álvarez",
    "Khéphren Thuram",
    "Joshua Kimmich",
    "João Palhinha",
    "Granit Xhaka",
]
MUST_LOW = [
    "Bruno Fernandes",
    "Jude Bellingham",
    "Florian Wirtz",
    "Thomas Müller",
    "Bukayo Saka",
]
BOUNDARY = ["Rúben Neves", "Joey Veerman", "Frenkie de Jong", "Pedri", "Sandro Tonali"]

# ---------------------------------------------------------------------------
# Feature weights (from FEATURES.md)
# ---------------------------------------------------------------------------

WEIGHTS = {
    # defensive volume (PAdj) — 1.5x
    "tackles_p90_padj": 1.5,
    "tackles_won_pct": 1.5,
    "interceptions_p90_padj": 1.5,
    "blocks_p90_padj": 1.5,
    "recoveries_p90_padj": 1.5,
    "aerials_won_pct": 1.5,
    # territory — 1.5x
    "pct_touches_def_third": 1.5,
    "pct_touches_mid_third": 1.5,
    "pct_touches_att_third": 1.5,
    # passing volume + accuracy — 1.0x
    "passes_total_p90": 1.0,
    "pass_cmp_pct": 1.0,
    "short_cmp_pct": 1.0,
    "medium_cmp_pct": 1.0,
    "long_cmp_pct": 1.0,
    "passes_per_touch": 1.0,
    # progression — 1.0x
    "prog_passes_p90": 1.0,
    "prog_carries_p90": 1.0,
    "passes_final_third_p90": 1.0,
    "passes_pen_area_p90": 1.0,
    "switches_p90": 1.0,
    # creation — 0.5x
    "xa_p90": 0.5,
    "key_passes_p90": 0.5,
    "sca_p90": 0.5,
    # shot output — sign-flipped, 1.0x
    "shots_p90_flip": 1.0,
    "npxg_p90_flip": 1.0,
    # ball security — sign-flipped, 1.0x
    "miscontrols_p90_flip": 1.0,
    "dispossessed_p90_flip": 1.0,
}

FEATURE_COLS = list(WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Column finder — maps logical feature -> actual column in each df
# Populated after --inspect run; fill these in after you run --inspect
# ---------------------------------------------------------------------------

# These are best-guess mappings; --inspect will show actual column names
COL_MAP = {
    # from 'passing'
    "90s": ("passing", "90s"),
    "passes_att": ("passing", "Total_Att"),
    "pass_cmp_pct": ("passing", "Total_Cmp%"),
    "short_cmp_pct": ("passing", "Short_Cmp%"),
    "medium_cmp_pct": ("passing", "Medium_Cmp%"),
    "long_cmp_pct": ("passing", "Long_Cmp%"),
    "prog_passes": ("passing", "PrgP"),
    "key_passes": ("passing", "KP"),
    "passes_final_third": ("passing", "1/3"),
    "passes_pen_area": ("passing", "PPA"),
    "xa": ("passing", "xAG"),
    # from 'pass types'
    "switches": ("pass types", "Sw"),
    # from 'defensive'
    "tackles": ("defensive", "Tackles_Tkl"),
    "tackles_won": ("defensive", "Tackles_TklW"),
    "interceptions": ("defensive", "Int"),
    "blocks": ("defensive", "Blocks_Blocks"),
    # from 'possession'
    "touches": ("possession", "Touches_Touches"),
    "touches_def_3rd": ("possession", "Touches_Def 3rd"),
    "touches_mid_3rd": ("possession", "Touches_Mid 3rd"),
    "touches_att_3rd": ("possession", "Touches_Att 3rd"),
    "prog_carries": ("possession", "Carries_PrgC"),
    "miscontrols": ("possession", "Miscontrols_Mis"),
    "dispossessed": ("possession", "Miscontrols_Dis"),
    # from 'goal and shot creation'
    "sca": ("goal and shot creation", "SCA_SCA"),
    # from 'shooting'
    "shots": ("shooting", "Standard_Sh"),
    "npxg": ("shooting", "Expected_npxG"),
    # from 'misc'
    "recoveries": ("misc", "Performance_Recov"),
    "aerials_won": ("misc", "Aerial Duels_Won"),
    "aerials_total": ("misc", "Aerial Duels_Won%"),  # may be ratio already
    # from 'standard'
    "minutes": ("standard", "Playing Time_Min"),
    "pos": ("standard", "pos"),
}


# ---------------------------------------------------------------------------
# Load all cached stat types
# ---------------------------------------------------------------------------


def load_all() -> dict[str, pd.DataFrame]:
    stat_types = [
        "passing",
        "pass types",
        "defensive",
        "possession",
        "goal and shot creation",
        "shooting",
        "misc",
    ]
    dfs = {}
    for st in stat_types:
        p = CACHE_DIR / f"combined__{st}.parquet"
        if not p.exists():
            print(f"  MISSING cache: {p} — run data_loader.py --full first")
            continue
        dfs[st] = pd.read_parquet(p)
        print(f"  Loaded '{st}': {dfs[st].shape}")
    return dfs


# ---------------------------------------------------------------------------
# Inspect columns
# ---------------------------------------------------------------------------


def inspect(dfs: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print("COLUMN INSPECTION — actual columns per stat type")
    print("=" * 60)
    for st, df in dfs.items():
        print(f"\n--- {st} ---")
        for c in sorted(df.columns):
            print(f"  {repr(c)}")


# ---------------------------------------------------------------------------
# Find actual column name (fuzzy fallback)
# ---------------------------------------------------------------------------


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy: check if candidate is a suffix
    for c in candidates:
        matches = [col for col in df.columns if col.endswith(c) or col == c]
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Main fingerprint builder
# ---------------------------------------------------------------------------


def build_fingerprint(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all stat types, filter midfielders, aggregate across seasons,
    compute features, normalize. Returns one row per player with z-scored features.
    """

    # --- Step 1: Load standard for minutes + position ---
    # We need minutes from passing (90s col) since standard isn't in our pull
    # Use passing as the base frame
    base = dfs["passing"].copy()

    # Identify columns — print what we have and fail loud on missing
    print("\n--- Base frame (passing) columns ---")
    for c in sorted(base.columns):
        print(f"  {repr(c)}")

    # Find key identity columns
    player_col = find_col(base, ["player", "Unnamed: 1_level_0_Player"])
    team_col = find_col(base, ["team", "Unnamed: 4_level_0_Squad"])
    pos_col = find_col(base, ["pos", "Unnamed: 3_level_0_Pos"])
    nineties = find_col(base, ["90s", "Unnamed: 7_level_0_90s"])

    print(
        f"\nIdentity cols: player={player_col}, team={team_col}, pos={pos_col}, 90s={nineties}"
    )

    if not all([player_col, team_col, pos_col, nineties]):
        raise RuntimeError(
            "Could not find identity columns in passing data. Run --inspect."
        )

    # Standardise identity col names in base
    base = base.rename(
        columns={
            player_col: "player",
            team_col: "team",
            pos_col: "pos",
            nineties: "90s",
        }
    )

    # Keep only columns we need from base
    passing_feature_cols = [
        c
        for c in [
            "Total_Att",
            "Total_Cmp%",
            "Short_Cmp%",
            "Medium_Cmp%",
            "Long_Cmp%",
            "PrgP",
            "KP",
            "xAG",
        ]
        if c in base.columns
    ]

    # Also check alternate column names
    for alt, canon in [
        ("Unnamed: 25_level_0_1/3", "passes_final_third_raw"),
        ("Unnamed: 26_level_0_PPA", "passes_pen_area_raw"),
        ("Unnamed: 24_level_0_KP", "KP"),
        ("Unnamed: 23_level_0_A-xAG", "xAG"),
    ]:
        if alt in base.columns and canon not in base.columns:
            base[canon] = base[alt]

    base["90s"] = pd.to_numeric(base["90s"], errors="coerce")
    base["minutes"] = base["90s"] * 90

    # --- Step 2: Filter midfielders ---
    # Primary pos == MF (first entry before comma)
    base["primary_pos"] = base["pos"].astype(str).str.split(",").str[0].str.strip()
    mf_mask = base["primary_pos"] == "MF"
    base = base[mf_mask].copy()
    print(f"\nAfter MF filter: {len(base)} rows")

    # --- Step 3: Merge other stat types onto base ---
    merge_keys = ["player", "team", "league", "season"]

    def merge_stat(base_df, stat_df, stat_type):
        # Standardise identity cols in stat_df
        pc = find_col(stat_df, ["player", "Unnamed: 1_level_0_Player"])
        tc = find_col(stat_df, ["team", "Unnamed: 4_level_0_Squad"])

        if pc and pc != "player":
            stat_df = stat_df.rename(columns={pc: "player"})
        if tc and tc != "team":
            stat_df = stat_df.rename(columns={tc: "team"})

        # Drop columns that already exist in base (except merge keys)
        overlap = [
            c for c in stat_df.columns if c in base_df.columns and c not in merge_keys
        ]
        stat_df = stat_df.drop(columns=overlap, errors="ignore")

        merged = base_df.merge(
            stat_df, on=merge_keys, how="left", suffixes=("", f"_{stat_type[:3]}")
        )
        print(f"  After merging '{stat_type}': {merged.shape}")
        return merged

    for st in [
        "pass types",
        "defensive",
        "possession",
        "goal and shot creation",
        "shooting",
        "misc",
    ]:
        if st in dfs:
            base = merge_stat(base, dfs[st].copy(), st)

    # --- Step 4: Minutes-weighted aggregation across seasons ---
    # Group by (player, team, league) — aggregate numerics by weighted mean
    print(f"\nPre-aggregation shape: {base.shape}")

    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [
        "player",
        "team",
        "league",
        "pos",
        "primary_pos",
        "nation",
        "season",
        "stat_type",
        "age",
        "born",
        "rk",
        "Player ID",
    ]

    # Minutes-weighted mean
    def wmean(group):
        mins = group["minutes"].fillna(0)
        total_mins = mins.sum()
        result = {}
        for c in numeric_cols:
            if c == "minutes":
                result[c] = total_mins
            elif c == "90s":
                result[c] = total_mins / 90
            else:
                vals = group[c].fillna(0)
                result[c] = (
                    (vals * mins).sum() / total_mins if total_mins > 0 else np.nan
                )
        # Keep metadata from most recent season
        result["team"] = group.sort_values("season").iloc[-1]["team"]
        result["league"] = group.sort_values("season").iloc[-1]["league"]
        result["pos"] = group.sort_values("season").iloc[-1]["pos"]
        return pd.Series(result)

    agg = base.groupby("player").apply(wmean).reset_index()
    print(f"Post-aggregation (one row per player): {agg.shape}")

    # --- Step 5: Minutes threshold ---
    agg = agg[agg["minutes"] >= 1500].copy()
    print(f"After 1500-min threshold: {len(agg)} players")

    # --- Step 6: Compute per-90 features ---
    nineties_agg = agg["90s"].clip(lower=0.01)  # avoid div by zero

    def p90(col):
        if col in agg.columns:
            return agg[col] / nineties_agg
        return np.nan

    # Find actual column names (may vary) — try candidates
    def get(candidates):
        for c in candidates:
            if c in agg.columns:
                return agg[c]
        return pd.Series(np.nan, index=agg.index)

    # PASSING
    agg["passes_total_p90"] = get(["Total_Att"]) / nineties_agg
    agg["pass_cmp_pct"] = get(["Total_Cmp%"])
    agg["short_cmp_pct"] = get(["Short_Cmp%"])
    agg["medium_cmp_pct"] = get(["Medium_Cmp%"])
    agg["long_cmp_pct"] = get(["Long_Cmp%"])
    agg["prog_passes_p90"] = get(["PrgP"]) / nineties_agg  # from pass types
    agg["key_passes_p90"] = get(["Unnamed: 24_level_0_KP"]) / nineties_agg
    agg["passes_final_third_p90"] = get(["Unnamed: 25_level_0_1/3"]) / nineties_agg
    agg["passes_pen_area_p90"] = get(["Unnamed: 26_level_0_PPA"]) / nineties_agg
    agg["xa_p90"] = get(["Unnamed: 23_level_0_A-xAG"]) / nineties_agg
    agg["switches_p90"] = get(["Pass Types_Sw"]) / nineties_agg  # from pass types
    # needs pass types pull

    # TOUCHES
    touches = get(["Touches_Touches"])
    agg["passes_per_touch"] = get(["Total_Att"]) / touches.clip(lower=1)
    agg["pct_touches_def_third"] = get(["Touches_Def 3rd"]) / touches.clip(lower=1)
    agg["pct_touches_mid_third"] = get(["Touches_Mid 3rd"]) / touches.clip(lower=1)
    agg["pct_touches_att_third"] = get(["Touches_Att 3rd"]) / touches.clip(lower=1)
    agg["prog_carries_p90"] = get(["Carries_PrgDist"]) / nineties_agg
    agg["miscontrols_p90"] = get(["Carries_Mis"]) / nineties_agg
    agg["dispossessed_p90"] = get(["Carries_Dis"]) / nineties_agg

    # DEFENSIVE
    agg["tackles_raw_p90"] = get(["Tackles_Tkl"]) / nineties_agg
    agg["tackles_won_pct"] = get(["Tackles_TklW"]) / get(["Tackles_Tkl"]).clip(lower=1)
    agg["interceptions_p90"] = get(["Unnamed: 20_level_0_Int"]) / nineties_agg
    agg["blocks_p90"] = get(["Blocks_Blocks"]) / nineties_agg
    agg["recoveries_p90"] = pd.Series(np.nan, index=agg.index)  # not in misc
    agg["aerials_won_pct"] = pd.Series(np.nan, index=agg.index)  # not in misc

    # GCA/SCA
    agg["sca_p90"] = get(["SCA_SCA"]) / nineties_agg

    # SHOOTING
    agg["shots_p90"] = get(["Standard_Sh"]) / nineties_agg
    agg["npxg_p90"] = pd.Series(np.nan, index=agg.index)  # not in shooting

    # --- Step 7: PAdj for defensive features ---
    # Using a fixed approximation: average team possession in top leagues ~52%
    # Proper PAdj = raw_p90 / (1 - team_poss_share)
    # Without per-player team possession data, use 0.52 as the mean
    # This normalizes out the possession effect on a population level
    # TODO: replace with actual team possession % from FBref team stats
    OPP_POSS = 0.48  # approximate opponent possession share
    for raw, padj in [
        ("tackles_raw_p90", "tackles_p90_padj"),
        ("interceptions_p90", "interceptions_p90_padj"),
        ("blocks_p90", "blocks_p90_padj"),
        ("recoveries_p90", "recoveries_p90_padj"),
    ]:
        if raw in agg.columns:
            agg[padj] = agg[raw] / OPP_POSS

    # --- Step 8: Sign-flip low-is-good features ---
    agg["shots_p90_flip"] = -agg["shots_p90"]
    agg["npxg_p90_flip"] = -agg["npxg_p90"]
    agg["miscontrols_p90_flip"] = -agg["miscontrols_p90"]
    agg["dispossessed_p90_flip"] = -agg["dispossessed_p90"]

    # --- Step 9: Z-score normalize ---
    available_features = [f for f in FEATURE_COLS if f in agg.columns]
    missing_features = [f for f in FEATURE_COLS if f not in agg.columns]
    if missing_features:
        print(f"\nMISSING FEATURES (will be NaN): {missing_features}")

    for f in available_features:
        mu = agg[f].mean()
        std = agg[f].std()
        agg[f"z_{f}"] = (agg[f] - mu) / std if std > 0 else 0.0

    z_cols = [f"z_{f}" for f in available_features]
    print(
        f"\nZ-scored {len(z_cols)} features. "
        f"{len(missing_features)} features missing (NaN in similarity)."
    )

    return agg, available_features


# ---------------------------------------------------------------------------
# Stage 1 — Similarity to seed vector
# ---------------------------------------------------------------------------


def run_stage1(agg: pd.DataFrame, available_features: list[str]) -> pd.DataFrame:
    z_cols = [f"z_{f}" for f in available_features]
    weights = np.array([WEIGHTS.get(f, 1.0) for f in available_features])

    # Build seed vector: mean of Rodri, Zubimendi, Tchouameni
    seed_names = [
        "Rodri",
        "Zubimendi",
        "Tchouaméni",
        "Martin Zubimendi",
        "Aurélien Tchouaméni",
    ]
    seeds = agg[agg["player"].isin(seed_names)]
    print(f"\nSeeds found: {seeds['player'].tolist()}")

    if len(seeds) == 0:
        print("WARNING: No seeds found in data. Check player name spelling.")
        print("Sample player names:", agg["player"].head(20).tolist())
        seed_vec = np.zeros(len(z_cols))
    else:
        seed_vec = seeds[z_cols].mean(axis=0).values

    # Weighted Euclidean distance
    Z = agg[z_cols].fillna(0).values
    diffs = (Z - seed_vec) * weights
    agg["archetype_distance"] = np.sqrt((diffs**2).sum(axis=1))
    agg["archetype_rank"] = agg["archetype_distance"].rank(method="min")

    result = agg[
        [
            "player",
            "team",
            "league",
            "pos",
            "minutes",
            "archetype_distance",
            "archetype_rank",
        ]
    ].copy()
    result = result.sort_values("archetype_rank")
    return result


# ---------------------------------------------------------------------------
# Validation check
# ---------------------------------------------------------------------------


def validate(ranked: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION (pre-registered in NOTES.md)")
    print("=" * 60)

    def find_rank(name):
        matches = ranked[
            ranked["player"].str.contains(name.split()[-1], case=False, na=False)
        ]
        if len(matches) == 0:
            return None
        return int(matches.iloc[0]["archetype_rank"])

    print("\nMust be top 10 (seeds):")
    for name in MUST_TOP_10:
        rank = find_rank(name)
        status = "✓" if rank and rank <= 10 else "✗ FAIL" if rank else "? NOT FOUND"
        print(f"  {status:8s} {name:30s} rank={rank}")

    print("\nMust be top 50:")
    for name in MUST_TOP_50:
        rank = find_rank(name)
        status = "✓" if rank and rank <= 50 else "✗ FAIL" if rank else "? NOT FOUND"
        print(f"  {status:8s} {name:30s} rank={rank}")

    print("\nMust rank low (negative controls):")
    for name in MUST_LOW:
        rank = find_rank(name)
        status = "✓" if rank and rank > 100 else "✗ FAIL" if rank else "? NOT FOUND"
        print(f"  {status:8s} {name:30s} rank={rank}")

    print("\nBoundary cases (no pass/fail):")
    for name in BOUNDARY:
        rank = find_rank(name)
        print(f"  {'?':8s} {name:30s} rank={rank}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print all columns per stat type then exit",
    )
    parser.add_argument(
        "--run", action="store_true", help="Build fingerprint and run Stage 1"
    )
    args = parser.parse_args()

    print("Loading cached data...")
    dfs = load_all()

    if args.inspect:
        inspect(dfs)
        print("\nRun with --run after reviewing columns above.")

    elif args.run:
        agg, available_features = build_fingerprint(dfs)
        agg.to_parquet(OUT_DIR / "fingerprint.parquet", index=False)
        print(f"\nSaved fingerprint: {OUT_DIR}/fingerprint.parquet")

        ranked = run_stage1(agg, available_features)
        ranked.to_parquet(OUT_DIR / "stage1_ranked.parquet", index=False)

        print("\n=== TOP 30 CANDIDATES ===")
        print(ranked.head(30).to_string(index=False))

        validate(ranked)

    else:
        print("Specify --inspect or --run")
        print("Start with: python src/fingerprint.py --inspect")
