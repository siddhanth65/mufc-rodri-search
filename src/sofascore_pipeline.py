"""
sofascore_pipeline.py v2 — Refined Stage 1.

Changes from v1:
  - Adds 25/26 season alongside 24/25 and 23/24
  - Expands seed set to 5 archetypal #6s (Rodri, Rice, Zubimendi, Tchouameni, Lobotka)
  - Switches similarity from "distance to seed mean" to "min distance to any seed"
    (handles multi-modal archetype)
  - Splits output into MAIN track (>=2000 mins) and PROSPECT track (600-2000 mins)
  - Re-runs seed variance check on expanded seed set

Usage:
  python src/sofascore_pipeline.py --pull     # pull 25/26 (24/25 and 23/24 already cached)
  python src/sofascore_pipeline.py --run      # build fingerprint + rankings
  python src/sofascore_pipeline.py --all      # pull then run
"""

from __future__ import annotations
import argparse
from difflib import SequenceMatcher
import io
import sys
import time
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so accented player names print.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from ScraperFC import Sofascore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEAGUES = [
    "England Premier League",
    "Spain La Liga",
    "Germany Bundesliga",
    "Italy Serie A",
    "France Ligue 1",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
]
SEASONS = ["25/26", "24/25", "23/24"]

MIN_MINUTES_MAIN = 2000  # main track: established starters
MIN_MINUTES_PROSPECT = 600  # prospect track: rising talent, lower sample

RATE_LIMIT = 5

CACHE_DIR = Path("data/sofascore")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seeds — the archetype anchors
# Spans destroyer (Tchouaméni) → all-court (Rodri, Rice) → deep-progressor
# (Zubimendi, Lobotka). Min-distance-to-any-seed handles the spread.
# ---------------------------------------------------------------------------

SEED_NAMES = [
    "Rodri",
    "Declan Rice",
    "Martín Zubimendi",
    "Martin Zubimendi",
    "Aurélien Tchouaméni",
    "Aurelien Tchouameni",
]

# ---------------------------------------------------------------------------
# Validation set (pre-registered)
# Note: Rice, Lobotka are now seeds — they MUST be top 10 by definition
# ---------------------------------------------------------------------------

MUST_TOP_10 = [
    "Rodri",
    "Martín Zubimendi",
    "Aurélien Tchouaméni",
    "Declan Rice",
    "Stanislav Lobotka",
]
MUST_TOP_50 = [
    "Moisés Caicedo",
    "Angelo Stiller",
    "Edson Álvarez",
    "Khéphren Thuram",
    "Joshua Kimmich",
    "João Palhinha",
    "Granit Xhaka",
]
MUST_LOW = ["Bruno Fernandes", "Jude Bellingham", "Florian Wirtz", "Thomas Müller"]
BOUNDARY = [
    "Rúben Neves",
    "Joey Veerman",
    "Frenkie de Jong",
    "Pedri",
    "Sandro Tonali",
    "Carlos Baleba",
    "Matheus Fernandes",
    "Elliot Anderson",
    "Ayoub Bouaddi",
    "Lamine Camara",
    "Boubacar Kamara",
]

# ---------------------------------------------------------------------------
# Feature spec (same as v1)
# ---------------------------------------------------------------------------

FEATURES = [
    ("passes_total", "totalPasses", "count"),
    ("pass_cmp_pct", "accuratePassesPercentage", "rate"),
    ("long_ball_cmp_pct", "accurateLongBallsPercentage", "rate"),
    ("passes_final_third", "accurateFinalThirdPasses", "count"),
    ("own_half_passes", "totalOwnHalfPasses", "count"),
    ("tackles", "tackles", "count"),
    ("tackles_won_pct", "tacklesWonPercentage", "rate"),
    ("interceptions", "interceptions", "count"),
    ("blocks", "outfielderBlocks", "count"),
    ("recoveries", "ballRecovery", "count"),
    ("aerials_won_pct", "aerialDuelsWonPercentage", "rate"),
    ("xa", "expectedAssists", "count"),
    ("key_passes", "keyPasses", "count"),
    ("shots", "totalShots", "count_flip"),
    ("xg", "expectedGoals", "count_flip"),
    ("dispossessed", "dispossessed", "count_flip"),
    ("possession_lost", "possessionLost", "count_flip"),
    ("touches", "touches", "count"),
    ("aerial_duels_won", "aerialDuelsWon", "count"),
    ("clearances", "clearances", "count"),
]

WEIGHTS = {
    "passes_total": 1.0,
    "pass_cmp_pct": 1.0,
    "long_ball_cmp_pct": 1.0,
    "passes_final_third": 1.0,
    "own_half_passes": 1.5,
    "tackles": 1.5,
    "tackles_won_pct": 1.5,
    "interceptions": 1.5,
    "blocks": 1.0,
    "recoveries": 1.5,
    "aerials_won_pct": 1.0,
    "xa": 0.5,
    "key_passes": 0.5,
    "shots": 1.0,
    "xg": 1.0,
    "dispossessed": 1.0,
    "possession_lost": 1.0,
    "touches": 1.0,
    "aerial_duels_won": 1.0,
    "clearances": 1.0,
}

OPP_POSS = 0.48
PADJ_FEATURES = {"tackles", "interceptions", "blocks", "recoveries"}

# Defensive ball-winning: exceeding the seed is free (more is monotonically good
# for a #6). Only shortfall below the seed is penalised.
ONE_SIDED_FEATURES = {"tackles", "interceptions", "recoveries", "blocks"}


# ---------------------------------------------------------------------------
# Step 1: Pull
# ---------------------------------------------------------------------------


def pull_all(force_refresh: bool = False) -> None:
    s = Sofascore()
    total = len(LEAGUES) * len(SEASONS)
    i = 0

    for league in LEAGUES:
        for season in SEASONS:
            i += 1
            safe = f"{league.replace(' ','_')}__{season.replace('/','_')}.parquet"
            cache = CACHE_DIR / safe

            if cache.exists() and not force_refresh:
                print(f"[{i}/{total}] Cache hit: {safe}")
                continue

            print(f"[{i}/{total}] Pulling {league} {season}...", end=" ", flush=True)
            try:
                df = s.scrape_player_league_stats(
                    year=season,
                    league=league,
                    accumulation="total",
                    selected_positions=["Midfielders"],
                )
                df["league"] = league
                df["season"] = season
                df.to_parquet(cache, index=False)
                print(f"OK ({len(df)} rows)")
            except Exception as e:
                print(f"FAILED: {e}")

            if i < total:
                time.sleep(RATE_LIMIT)


# ---------------------------------------------------------------------------
# Step 2: Load + aggregate
# ---------------------------------------------------------------------------


def load_and_aggregate() -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        for season in SEASONS:
            safe = f"{league.replace(' ','_')}__{season.replace('/','_')}.parquet"
            cache = CACHE_DIR / safe
            if cache.exists():
                frames.append(pd.read_parquet(cache))
            else:
                print(f"  MISSING: {safe}")
    if not frames:
        raise RuntimeError("No cached data. Run --pull first.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(raw)} rows across {len(frames)} league-seasons")

    id_cols = {"player", "team", "league", "season", "player id", "team id"}
    for col in raw.columns:
        if col not in id_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    count_cols = [f[1] for f in FEATURES if f[2] in ("count", "count_flip")]
    rate_cols = [f[1] for f in FEATURES if f[2] == "rate"]

    records = []
    for player_name, grp in raw.groupby("player"):
        total_min = grp["minutesPlayed"].sum()
        if total_min == 0:
            continue
        rec = {
            "player": player_name,
            "team": grp.sort_values("season").iloc[-1]["team"],
            "league": grp.sort_values("season").iloc[-1]["league"],
            "minutes": total_min,
            "90s": total_min / 90,
            "seasons_played": grp["season"].nunique(),
        }
        for col in count_cols:
            rec[col] = grp[col].sum() if col in grp.columns else np.nan
        for col in rate_cols:
            if col in grp.columns:
                mins = grp["minutesPlayed"].fillna(0)
                vals = grp[col].fillna(0)
                rec[col] = (vals * mins).sum() / total_min if total_min > 0 else np.nan
            else:
                rec[col] = np.nan
        records.append(rec)

    agg = pd.DataFrame(records)
    print(f"Aggregated: {len(agg)} unique players")
    return agg


# ---------------------------------------------------------------------------
# Step 3: Fingerprint (z-scored within the universe of players we'll rank)
# ---------------------------------------------------------------------------


def build_fingerprint(agg: pd.DataFrame, ref_pool: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-90 features for agg.
    Z-score using mu/std from ref_pool (so main and prospect tracks share scale).
    """
    nineties = agg["90s"].clip(lower=0.01)
    for logical, sofa_col, ftype in FEATURES:
        if sofa_col not in agg.columns:
            agg[f"f_{logical}"] = np.nan
            continue
        if ftype == "count":
            agg[f"f_{logical}"] = agg[sofa_col] / nineties
        elif ftype == "count_flip":
            agg[f"f_{logical}"] = -(agg[sofa_col] / nineties)
        elif ftype == "rate":
            agg[f"f_{logical}"] = agg[sofa_col]

    for logical in PADJ_FEATURES:
        col = f"f_{logical}"
        if col in agg.columns:
            agg[col] = agg[col] / OPP_POSS

    # Compute the same on ref_pool to get mu/std
    ref_nineties = ref_pool["90s"].clip(lower=0.01)
    for logical, sofa_col, ftype in FEATURES:
        if sofa_col not in ref_pool.columns:
            continue
        if ftype == "count":
            ref_pool[f"f_{logical}"] = ref_pool[sofa_col] / ref_nineties
        elif ftype == "count_flip":
            ref_pool[f"f_{logical}"] = -(ref_pool[sofa_col] / ref_nineties)
        elif ftype == "rate":
            ref_pool[f"f_{logical}"] = ref_pool[sofa_col]
    for logical in PADJ_FEATURES:
        col = f"f_{logical}"
        if col in ref_pool.columns:
            ref_pool[col] = ref_pool[col] / OPP_POSS

    f_cols = [f"f_{f[0]}" for f in FEATURES if f"f_{f[0]}" in agg.columns]
    for col in f_cols:
        mu = ref_pool[col].mean()
        std = ref_pool[col].std()
        agg[f"z_{col[2:]}"] = (agg[col] - mu) / std if std > 0 else 0.0

    return agg


# ---------------------------------------------------------------------------
# Step 4: Stage 1 — min distance to any seed
# ---------------------------------------------------------------------------


def run_stage1(agg: pd.DataFrame, seeds_df: pd.DataFrame, label: str) -> pd.DataFrame:
    feat_order = [f[0] for f in FEATURES if f"z_{f[0]}" in agg.columns]
    z_cols     = [f"z_{f}" for f in feat_order]
    weights    = np.array([WEIGHTS.get(f, 1.0) for f in feat_order])
    one_sided  = np.array([f in ONE_SIDED_FEATURES for f in feat_order])

    if len(seeds_df) == 0:
        print(f"  No seeds in {label}. Skipping.")
        return None

    seed_Z = seeds_df[z_cols].fillna(0).values   # (n_seeds, n_features)
    Z      = agg[z_cols].fillna(0).values         # (n_players, n_features)

    # raw signed difference: (n_players, n_seeds, n_features)
    raw_diff = Z[:, None, :] - seed_Z[None, :, :]

    # One-sided: player above seed on defensive features => no penalty (clip to 0)
    adj_diff = raw_diff.copy()
    adj_diff[:, :, one_sided] = np.minimum(adj_diff[:, :, one_sided], 0.0)

    weighted = adj_diff * weights[None, None, :]
    dists    = np.sqrt((weighted ** 2).sum(axis=2))   # (n_players, n_seeds)

    agg["archetype_distance"] = dists.min(axis=1)
    agg["closest_seed_idx"]   = dists.argmin(axis=1)
    agg["closest_seed"]       = [seeds_df.iloc[i]["player"] for i in agg["closest_seed_idx"]]
    agg["archetype_rank"]     = agg["archetype_distance"].rank(method="min")

    ranked = agg[
        [
            "player",
            "team",
            "league",
            "minutes",
            "closest_seed",
            "archetype_distance",
            "archetype_rank",
        ]
    ].sort_values("archetype_rank")
    return ranked


# ---------------------------------------------------------------------------
# Seed variance check
# ---------------------------------------------------------------------------


def seed_variance_check(seeds_df: pd.DataFrame, agg: pd.DataFrame) -> None:
    z_cols = [f"z_{f[0]}" for f in FEATURES if f"z_{f[0]}" in seeds_df.columns]
    if len(seeds_df) < 2:
        return
    sZ = seeds_df[z_cols].fillna(0).values
    pairwise = []
    for i in range(len(sZ)):
        for j in range(i + 1, len(sZ)):
            d = np.sqrt(((sZ[i] - sZ[j]) ** 2).sum())
            pairwise.append((seeds_df.iloc[i]["player"], seeds_df.iloc[j]["player"], d))
    pZ = agg[z_cols].fillna(0).values
    pop_med = np.median(
        [
            np.sqrt(((pZ[i] - pZ[j]) ** 2).sum())
            for i in range(min(200, len(pZ)))
            for j in range(i + 1, min(200, len(pZ)))
        ]
    )
    print(f"\nPopulation median pairwise distance: {pop_med:.2f}")
    print("Seed pairwise distances:")
    for a, b, d in sorted(pairwise, key=lambda x: x[2]):
        flag = " (tight)" if d < pop_med else " (LOOSE)"
        print(f"  {a:25s} <-> {b:25s} {d:5.2f}{flag}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(ranked: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    def find_rank(name):
        last = name.split()[-1]
        m = ranked[ranked["player"].str.contains(last, case=False, na=False)]
        if len(m) == 0:
            return None
        return int(m.iloc[0]["archetype_rank"])

    print("\nMust be top 10 (seeds — failure = bug):")
    for name in MUST_TOP_10:
        r = find_rank(name)
        status = "✓" if r and r <= 10 else ("✗ FAIL" if r else "? NOT FOUND")
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nMust be top 50:")
    for name in MUST_TOP_50:
        r = find_rank(name)
        status = "✓" if r and r <= 50 else ("✗ FAIL" if r else "? NOT FOUND")
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nMust rank below top 100 (negative controls):")
    for name in MUST_LOW:
        r = find_rank(name)
        status = "✓" if r and r > 100 else ("✗ FAIL" if r else "? NOT FOUND")
        print(f"  {status:10s} {name:30s} rank={r}")

    print("\nBoundary cases / scout interest (no pass/fail):")
    for name in BOUNDARY:
        r = find_rank(name)
        print(f"  {'?':10s} {name:30s} rank={r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Add this to src/sofascore_pipeline.py — after the validate() function,
# before the "if __name__ == ..." block


def _normalize_player_name(name: str) -> str:
    return " ".join(str(name).lower().split())


def _find_player_for_diagnosis(
    agg: pd.DataFrame, name: str
) -> tuple[pd.Series | None, pd.DataFrame]:
    target = _normalize_player_name(name)
    player_names = agg["player"].fillna("").map(_normalize_player_name)

    exact_matches = agg[player_names == target]
    if len(exact_matches) > 0:
        return exact_matches.sort_values("minutes", ascending=False).iloc[0], exact_matches

    full_matches = agg[player_names.str.contains(target, regex=False, na=False)]
    if len(full_matches) > 0:
        return full_matches.sort_values("minutes", ascending=False).iloc[0], full_matches

    last = target.split()[-1]
    matches = agg[player_names.str.contains(last, regex=False, na=False)]
    if len(matches) == 0:
        return None, matches

    scores = matches["player"].map(
        lambda player: SequenceMatcher(
            None, target, _normalize_player_name(player)
        ).ratio()
    )
    ranked_matches = (
        matches.assign(_name_match_score=scores)
        .sort_values(["_name_match_score", "minutes"], ascending=[False, False])
        .drop(columns="_name_match_score")
    )
    return ranked_matches.iloc[0], ranked_matches


def diagnose_player(agg: pd.DataFrame, seeds_df: pd.DataFrame, name: str) -> None:
    """
    Show why a player ranks where they do.
    Prints their feature vector, distance to each seed, and the 5 features
    that contribute most to their distance from the nearest seed.
    """
    feat_names = [f[0] for f in FEATURES if f"z_{f[0]}" in agg.columns]
    z_cols     = [f"z_{f}" for f in feat_names]
    weights    = np.array([WEIGHTS.get(f, 1.0) for f in feat_names])
    one_sided  = np.array([f in ONE_SIDED_FEATURES for f in feat_names])

    # Find player, preferring exact/full-name matches before surname fallback.
    player_row, matches = _find_player_for_diagnosis(agg, name)
    if player_row is None:
        print(f"  '{name}' not found")
        return
    if len(matches) > 1:
        print(f"  Multiple matches for '{name}':")
        for _, r in matches.iterrows():
            print(f"    {r['player']:30s} {r['team']:25s} mins={int(r['minutes'])}")
        print(f"  Using: {player_row['player']} ({player_row['team']})")

    p_vec = player_row[z_cols].fillna(0).to_numpy(dtype=float)
    print(f"\n{'='*60}")
    print(
        f"DIAGNOSE: {player_row['player']} ({player_row['team']}, "
        f"{int(player_row['minutes'])} mins)"
    )
    print(f"Overall rank: {int(player_row.get('archetype_rank', -1))}")
    print(f"{'='*60}")

    # Distance to each seed
    print(f"\nDistance to each seed:")
    seed_distances = []
    for _, seed in seeds_df.iterrows():
        s_vec = seed[z_cols].fillna(0).to_numpy(dtype=float)
        raw   = p_vec - s_vec
        adj   = raw.copy()
        adj[one_sided] = np.minimum(adj[one_sided], 0.0)
        d = np.sqrt((adj ** 2 * weights ** 2).sum())
        seed_distances.append((seed["player"], d, s_vec))
    seed_distances.sort(key=lambda x: x[1])
    for sname, d, _ in seed_distances:
        print(f"  {sname:30s} {d:6.2f}")

    # Top features driving distance from CLOSEST seed
    closest_name, closest_d, closest_vec = seed_distances[0]
    raw           = p_vec - closest_vec
    adj           = raw.copy()
    adj[one_sided] = np.minimum(adj[one_sided], 0.0)
    feature_diffs = adj ** 2 * weights ** 2
    sorted_idx = np.argsort(feature_diffs)[::-1]

    print(f"\nTop 8 features driving distance from {closest_name}:")
    print(f"  {'feature':25s} {'player_z':>10s} {'seed_z':>10s} {'weighted_diff':>14s}")
    for i in sorted_idx[:8]:
        fname = feat_names[i]
        print(
            f"  {fname:25s} {p_vec[i]:>10.2f} {closest_vec[i]:>10.2f} "
            f"{feature_diffs[i]:>14.2f}"
        )

    # Show raw feature values vs seed for top distinguishers
    print(f"\nRaw values (per 90 unless rate):")
    raw_cols = [
        f"f_{feat_names[i]}"
        for i in sorted_idx[:8]
        if f"f_{feat_names[i]}" in agg.columns
    ]
    comp = pd.DataFrame(
        {
            player_row["player"]: [
                player_row[c] if c in player_row.index else np.nan for c in raw_cols
            ],
            closest_name: [
                (
                    seeds_df[seeds_df["player"] == closest_name].iloc[0][c]
                    if c in seeds_df.columns
                    else np.nan
                )
                for c in raw_cols
            ],
        },
        index=[c[2:] for c in raw_cols],
    )
    print(comp.to_string())


# Then in the entry point, add a --diagnose argument:
#
# parser.add_argument("--diagnose", nargs="+", default=None,
#                     help="Player name(s) to diagnose")
#
# And handle it:
# if args.diagnose:
#     agg = load_and_aggregate()
#     main = agg[agg["minutes"] >= MIN_MINUTES_MAIN].copy()
#     prospect = agg[(agg["minutes"] >= MIN_MINUTES_PROSPECT)
#                    & (agg["minutes"] < MIN_MINUTES_MAIN)].copy()
#     all_players = pd.concat([main, prospect], ignore_index=True)
#     all_players = build_fingerprint(all_players, ref_pool=main.copy())
#     seeds = main.copy()
#     seeds = build_fingerprint(seeds, ref_pool=main.copy())
#     seeds = seeds[seeds["player"].isin(SEED_NAMES)].drop_duplicates("player")
#     # Need to also compute archetype_rank for context
#     ranked = run_stage1(all_players.copy(), seeds, "all")
#     all_players = all_players.merge(
#         ranked[["player", "archetype_rank"]], on="player", how="left"
#     )
#     for name in args.diagnose:
#         diagnose_player(all_players, seeds, name)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnose", nargs="+", default=None, help="Player name(s) to diagnose"
    )
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.pull or args.all:
        print(
            f"Pulling {len(LEAGUES)} leagues x {len(SEASONS)} seasons (25/26 + cached)\n"
        )
        pull_all()
        print("\nPull complete.")

    if args.diagnose:
        agg = load_and_aggregate()
        main = agg[agg["minutes"] >= MIN_MINUTES_MAIN].copy().reset_index(drop=True)
        prospect = (
            agg[
                (agg["minutes"] >= MIN_MINUTES_PROSPECT)
                & (agg["minutes"] < MIN_MINUTES_MAIN)
            ]
            .copy()
            .reset_index(drop=True)
        )
        all_p = pd.concat([main, prospect], ignore_index=True)
        main_fp = build_fingerprint(main.copy(), ref_pool=main.copy())
        all_p = build_fingerprint(all_p, ref_pool=main.copy())
        seeds = main_fp[main_fp["player"].isin(SEED_NAMES)].drop_duplicates("player")
        ranked = run_stage1(all_p.copy(), seeds, "all")
        all_p = all_p.merge(
            ranked[["player", "archetype_rank"]], on="player", how="left"
        )
        for name in args.diagnose:
            diagnose_player(all_p, seeds, name)

    if args.run or args.all:
        agg = load_and_aggregate()

        # Split into main + prospect by minutes
        main = agg[agg["minutes"] >= MIN_MINUTES_MAIN].copy().reset_index(drop=True)
        prospect = (
            agg[
                (agg["minutes"] >= MIN_MINUTES_PROSPECT)
                & (agg["minutes"] < MIN_MINUTES_MAIN)
            ]
            .copy()
            .reset_index(drop=True)
        )

        print(f"\nMain track: {len(main)} players (>={MIN_MINUTES_MAIN} min)")
        print(
            f"Prospect track: {len(prospect)} players "
            f"({MIN_MINUTES_PROSPECT}-{MIN_MINUTES_MAIN} min)"
        )

        # Build fingerprint — z-score both tracks against main pool
        main = build_fingerprint(main.copy(), ref_pool=main.copy())
        prospect = build_fingerprint(prospect.copy(), ref_pool=main.copy())

        # Find seeds (must be in main pool)
        seeds = main[main["player"].isin(SEED_NAMES)].drop_duplicates("player")
        print(f"\nSeeds resolved ({len(seeds)}/{len(set(SEED_NAMES))} canonical):")
        for _, row in seeds.iterrows():
            print(f"  {row['player']:30s} {row['team']:25s} mins={int(row['minutes'])}")

        seed_variance_check(seeds, main)

        # MAIN ranking
        ranked_main = run_stage1(main, seeds, "main")
        # PROSPECT ranking — uses same seed vectors
        ranked_prospect = run_stage1(prospect, seeds, "prospect")

        # Save
        ranked_main.to_parquet(OUT_DIR / "ranked_main.parquet", index=False)
        if ranked_prospect is not None:
            ranked_prospect.to_parquet(OUT_DIR / "ranked_prospect.parquet", index=False)

        print("\n" + "=" * 60)
        print("MAIN TRACK — TOP 30")
        print("=" * 60)
        print(ranked_main.head(30).to_string(index=False))

        if ranked_prospect is not None:
            print("\n" + "=" * 60)
            print("PROSPECT TRACK — TOP 20 (600-2000 mins)")
            print("=" * 60)
            print(ranked_prospect.head(20).to_string(index=False))

        validate(ranked_main)
        print(f"\nSaved: {OUT_DIR}/ranked_main.parquet, ranked_prospect.parquet")

    if not any([args.pull, args.run, args.all, args.diagnose]):
        print("Specify --pull, --run, --diagnose, or --all")
