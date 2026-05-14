"""
fingerprint.py — v2: correct aggregation math + dedupe before merge.

KEY FIXES from v1:
  1. Drop duplicate (player, team, league, season) rows BEFORE merging
     (was inflating row count from 3386 -> 3449 due to duplicated stat rows)
  2. Sum-then-divide for volume features across seasons (NOT minutes-weighted mean)
     Correct math: sum(tackles_both_seasons) / sum(90s_both_seasons)
     v1 was computing weighted mean of season-level rates, which is wrong for raw counts
  3. Minutes-weighted mean ONLY for rate features (Cmp%, TklW%, possession-share %)
  4. Unicode name handling for Zubimendi etc.
"""

from __future__ import annotations
import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Validation set (pre-registered)
# ---------------------------------------------------------------------------

MUST_TOP_10 = ["Rodri", "Martín Zubimendi", "Aurélien Tchouaméni"]
MUST_TOP_50 = [
    "Declan Rice", "Moisés Caicedo", "Stanislav Lobotka",
    "Angelo Stiller", "Edson Álvarez", "Khéphren Thuram",
    "Joshua Kimmich", "João Palhinha", "Granit Xhaka",
]
MUST_LOW = [
    "Bruno Fernandes", "Jude Bellingham", "Florian Wirtz",
    "Thomas Müller", "Bukayo Saka",
]
BOUNDARY = ["Rúben Neves", "Joey Veerman", "Frenkie de Jong", "Pedri", "Sandro Tonali"]

# Seed names that may appear in data (handle unicode variants)
SEED_PATTERNS = ["Rodri", "Zubimendi", "Tchouaméni"]

WEIGHTS = {
    "tackles_p90_padj":       1.5,
    "tackles_won_pct":        1.5,
    "interceptions_p90_padj": 1.5,
    "blocks_p90_padj":        1.5,
    "pct_touches_def_third":  1.5,
    "pct_touches_mid_third":  1.5,
    "pct_touches_att_third":  1.5,
    "passes_total_p90":       1.0,
    "pass_cmp_pct":           1.0,
    "short_cmp_pct":          1.0,
    "medium_cmp_pct":         1.0,
    "long_cmp_pct":           1.0,
    "passes_per_touch":       1.0,
    "prog_passes_p90":        1.0,
    "prog_carries_p90":       1.0,
    "passes_final_third_p90": 1.0,
    "passes_pen_area_p90":    1.0,
    "switches_p90":           1.0,
    "xa_p90":                 0.5,
    "key_passes_p90":         0.5,
    "sca_p90":                0.5,
    "shots_p90_flip":         1.0,
    "miscontrols_p90_flip":   1.0,
    "dispossessed_p90_flip":  1.0,
}
FEATURE_COLS = list(WEIGHTS.keys())


# Which raw inputs are "volume counts" (sum-then-divide across seasons) vs "rates" (weighted mean)
VOLUME_COLS = [
    "Total_Att", "Unnamed: 24_level_0_KP", "Unnamed: 25_level_0_1/3",
    "Unnamed: 26_level_0_PPA", "Unnamed: 23_level_0_A-xAG",
    "PrgP", "Sw",
    "Tackles_Tkl", "Tackles_TklW", "Unnamed: 20_level_0_Int", "Blocks_Blocks",
    "Touches_Touches", "Touches_Def 3rd", "Touches_Mid 3rd", "Touches_Att 3rd",
    "Carries_PrgDist", "Carries_Mis", "Carries_Dis",
    "SCA_SCA",
    "Standard_Sh",
]

RATE_COLS = [
    "Total_Cmp%", "Short_Cmp%", "Medium_Cmp%", "Long_Cmp%",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    stat_types = [
        "passing", "pass types", "defensive", "possession",
        "goal and shot creation", "shooting", "misc",
    ]
    dfs = {}
    for st in stat_types:
        p = CACHE_DIR / f"combined__{st}.parquet"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        dfs[st] = pd.read_parquet(p)
        print(f"  Loaded '{st}': {dfs[st].shape}")
    return dfs


def inspect(dfs):
    for st, df in dfs.items():
        print(f"\n--- {st} ---")
        for c in sorted(df.columns):
            print(f"  {repr(c)}")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def dedupe(df, name):
    """Drop duplicate (player, team, league, season) rows. Keep first."""
    before = len(df)
    df = df.drop_duplicates(subset=["player", "team", "league", "season"], keep="first")
    after = len(df)
    if before != after:
        print(f"  Deduped {name}: {before} -> {after} ({before - after} dupes removed)")
    return df


# ---------------------------------------------------------------------------
# Build fingerprint
# ---------------------------------------------------------------------------

def build_fingerprint(dfs):
    base = dfs["passing"].copy()

    pc = find_col(base, ["player", "Unnamed: 1_level_0_Player"])
    tc = find_col(base, ["team",   "Unnamed: 4_level_0_Squad"])
    poc = find_col(base, ["pos",   "Unnamed: 3_level_0_Pos"])
    nc  = find_col(base, ["90s",   "Unnamed: 7_level_0_90s"])

    base = base.rename(columns={pc: "player", tc: "team", poc: "pos", nc: "90s"})
    base["90s"] = pd.to_numeric(base["90s"], errors="coerce")
    base["minutes"] = base["90s"] * 90

    # Filter midfielders BEFORE merging — much cleaner
    base["primary_pos"] = base["pos"].astype(str).str.split(",").str[0].str.strip()
    base = base[base["primary_pos"] == "MF"].copy()
    print(f"\nAfter MF filter: {len(base)} rows")

    base = dedupe(base, "passing")

    # Merge other stat types, deduping each FIRST
    merge_keys = ["player", "team", "league", "season"]

    for st in ["pass types", "defensive", "possession",
               "goal and shot creation", "shooting", "misc"]:
        if st not in dfs:
            continue
        right = dfs[st].copy()

        # Standardise identity cols
        rpc = find_col(right, ["player", "Unnamed: 1_level_0_Player"])
        rtc = find_col(right, ["team",   "Unnamed: 4_level_0_Squad"])
        if rpc and rpc != "player":
            right = right.rename(columns={rpc: "player"})
        if rtc and rtc != "team":
            right = right.rename(columns={rtc: "team"})

        # Dedupe right side BEFORE merge — this fixes row inflation
        right = dedupe(right, st)

        # Drop overlap columns
        overlap = [c for c in right.columns
                   if c in base.columns and c not in merge_keys]
        right = right.drop(columns=overlap, errors="ignore")

        before = len(base)
        base = base.merge(right, on=merge_keys, how="left")
        print(f"  After merging '{st}': {len(base)} rows (was {before})")

    print(f"\nPre-aggregation: {base.shape}")

    # --- AGGREGATE ACROSS SEASONS ---
    # Volume cols: sum across seasons (then we divide by total 90s for per-90)
    # Rate cols:   minutes-weighted mean
    # 90s, minutes: sum

    def agg_player(g):
        result = {}
        total_90s = g["90s"].fillna(0).sum()
        total_min = g["minutes"].fillna(0).sum()
        result["90s"] = total_90s
        result["minutes"] = total_min

        for col in g.columns:
            if col in ("player", "team", "league", "pos", "primary_pos",
                       "nation", "season", "stat_type", "age", "born",
                       "rk", "Player ID", "90s", "minutes"):
                continue
            if col in VOLUME_COLS:
                # Sum across seasons
                result[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).sum()
            elif col in RATE_COLS:
                # Minutes-weighted mean
                vals = pd.to_numeric(g[col], errors="coerce").fillna(0)
                mins = g["90s"].fillna(0)
                result[col] = (vals * mins).sum() / mins.sum() if mins.sum() > 0 else np.nan
            else:
                # Default: minutes-weighted mean (safer than sum for unknown cols)
                vals = pd.to_numeric(g[col], errors="coerce").fillna(0)
                mins = g["90s"].fillna(0)
                result[col] = (vals * mins).sum() / mins.sum() if mins.sum() > 0 else np.nan

        result["team"]   = g.sort_values("season").iloc[-1]["team"]
        result["league"] = g.sort_values("season").iloc[-1]["league"]
        result["pos"]    = g.sort_values("season").iloc[-1]["pos"]
        return pd.Series(result)

    print("Aggregating across seasons (this takes ~30s)...")
    agg = base.groupby("player").apply(agg_player).reset_index()
    print(f"Post-aggregation: {agg.shape}")

    agg = agg[agg["minutes"] >= 1500].copy()
    print(f"After 1500-min threshold: {len(agg)} players")

    n90 = agg["90s"].clip(lower=0.01)

    def get(cands):
        for c in cands:
            if c in agg.columns:
                return pd.to_numeric(agg[c], errors="coerce")
        return pd.Series(np.nan, index=agg.index)

    # PASSING
    agg["passes_total_p90"]      = get(["Total_Att"]) / n90
    agg["pass_cmp_pct"]          = get(["Total_Cmp%"])
    agg["short_cmp_pct"]         = get(["Short_Cmp%"])
    agg["medium_cmp_pct"]        = get(["Medium_Cmp%"])
    agg["long_cmp_pct"]          = get(["Long_Cmp%"])
    agg["prog_passes_p90"]       = get(["PrgP"]) / n90
    agg["key_passes_p90"]        = get(["Unnamed: 24_level_0_KP"]) / n90
    agg["passes_final_third_p90"]= get(["Unnamed: 25_level_0_1/3"]) / n90
    agg["passes_pen_area_p90"]   = get(["Unnamed: 26_level_0_PPA"]) / n90
    agg["xa_p90"]                = get(["Unnamed: 23_level_0_A-xAG"]) / n90
    agg["switches_p90"]          = get(["Sw"]) / n90

    # TOUCHES
    touches = get(["Touches_Touches"]).clip(lower=1)
    agg["passes_per_touch"]        = get(["Total_Att"]) / touches
    agg["pct_touches_def_third"]   = get(["Touches_Def 3rd"]) / touches
    agg["pct_touches_mid_third"]   = get(["Touches_Mid 3rd"]) / touches
    agg["pct_touches_att_third"]   = get(["Touches_Att 3rd"]) / touches
    agg["prog_carries_p90"]        = get(["Carries_PrgDist"]) / n90
    agg["miscontrols_p90"]         = get(["Carries_Mis"]) / n90
    agg["dispossessed_p90"]        = get(["Carries_Dis"]) / n90

    # DEFENSIVE
    tkl = get(["Tackles_Tkl"])
    agg["tackles_raw_p90"]         = tkl / n90
    agg["tackles_won_pct"]         = get(["Tackles_TklW"]) / tkl.clip(lower=1)
    agg["interceptions_p90"]       = get(["Unnamed: 20_level_0_Int"]) / n90
    agg["blocks_p90"]              = get(["Blocks_Blocks"]) / n90

    # GCA/SCA
    agg["sca_p90"]                 = get(["SCA_SCA"]) / n90

    # SHOOTING
    agg["shots_p90"]               = get(["Standard_Sh"]) / n90

    # PAdj — approximate using 0.48 opp possession
    OPP_POSS = 0.48
    for raw, padj in [
        ("tackles_raw_p90",   "tackles_p90_padj"),
        ("interceptions_p90", "interceptions_p90_padj"),
        ("blocks_p90",        "blocks_p90_padj"),
    ]:
        agg[padj] = agg[raw] / OPP_POSS

    # Sign-flip low-is-good
    agg["shots_p90_flip"]        = -agg["shots_p90"]
    agg["miscontrols_p90_flip"]  = -agg["miscontrols_p90"]
    agg["dispossessed_p90_flip"] = -agg["dispossessed_p90"]

    available = [f for f in FEATURE_COLS if f in agg.columns]
    missing   = [f for f in FEATURE_COLS if f not in agg.columns]
    if missing:
        print(f"MISSING FEATURES: {missing}")

    # Z-score normalize
    for f in available:
        s = agg[f]
        mu = s.mean()
        std = s.std()
        agg[f"z_{f}"] = (s - mu) / std if std > 0 else 0.0

    print(f"Z-scored {len(available)} features.")
    return agg, available


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(agg, available):
    z_cols = [f"z_{f}" for f in available]
    weights = np.array([WEIGHTS.get(f, 1.0) for f in available])

    # Match seeds by substring (handles unicode)
    seed_mask = agg["player"].apply(
        lambda p: any(seed in str(p) for seed in SEED_PATTERNS)
    )
    seeds = agg[seed_mask]
    print(f"\nSeeds found: {seeds['player'].tolist()}")

    if len(seeds) == 0:
        raise RuntimeError("No seeds found.")

    seed_vec = seeds[z_cols].fillna(0).mean(axis=0).values

    Z = agg[z_cols].fillna(0).values
    diffs = (Z - seed_vec) * weights
    agg["archetype_distance"] = np.sqrt((diffs ** 2).sum(axis=1))
    agg["archetype_rank"] = agg["archetype_distance"].rank(method="min")

    result = agg[["player", "team", "league", "pos", "minutes",
                  "archetype_distance", "archetype_rank"]].copy()
    return result.sort_values("archetype_rank")


def validate(ranked):
    print("\n" + "="*60)
    print("VALIDATION (pre-registered in NOTES.md)")
    print("="*60)

    def find_rank(name):
        # Match by last word (surname), case-insensitive
        surname = name.split()[-1]
        matches = ranked[ranked["player"].str.contains(surname, case=False, na=False, regex=False)]
        if len(matches) == 0:
            return None
        return int(matches.iloc[0]["archetype_rank"])

    print("\nMust be top 10 (seeds):")
    for name in MUST_TOP_10:
        r = find_rank(name)
        status = "✓" if r and r <= 10 else "✗ FAIL" if r else "? NOT FOUND"
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nMust be top 50:")
    for name in MUST_TOP_50:
        r = find_rank(name)
        status = "✓" if r and r <= 50 else "✗ FAIL" if r else "? NOT FOUND"
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nMust rank below top 100:")
    for name in MUST_LOW:
        r = find_rank(name)
        status = "✓" if r and r > 100 else "✗ FAIL" if r else "? NOT FOUND"
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nBoundary cases:")
    for name in BOUNDARY:
        r = find_rank(name)
        print(f"  {'?':10s} {name:30s} rank={r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    print("Loading cached data...")
    dfs = load_all()

    if args.inspect:
        inspect(dfs)
    elif args.run:
        agg, available = build_fingerprint(dfs)
        agg.to_parquet(OUT_DIR / "fingerprint.parquet", index=False)
        ranked = run_stage1(agg, available)
        ranked.to_parquet(OUT_DIR / "stage1_ranked.parquet", index=False)

        print("\n=== TOP 30 ===")
        print(ranked.head(30).to_string(index=False))

        validate(ranked)
    else:
        print("Use --inspect or --run")