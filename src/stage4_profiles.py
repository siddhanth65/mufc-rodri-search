"""
stage4_profiles.py — Stage 4: Player profiles with percentile layer.

For each of the top 15 non-seed players from Stage 3, generates:
  - DataMB-style percentile table (8 key metrics vs main-track population)
  - Gap-fill summary (which Mainoo+Bruno weaknesses the player covers)
  - Auto-generated scouting note

Usage:
    python src/stage4_profiles.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sofascore_pipeline import (
    FEATURES,
    WEIGHTS,
    MIN_MINUTES_MAIN,
    MIN_MINUTES_PROSPECT,
    load_and_aggregate,
    build_fingerprint,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR      = Path("data/processed")
STAGE3_PATH  = OUT_DIR / "stage3_ranked.parquet"
RANKED_MAIN  = OUT_DIR / "ranked_main.parquet"

SEED_PLAYERS  = {"Rodri", "Declan Rice", "Martín Zubimendi", "Aurélien Tchouaméni"}
EXISTING_PAIR = ["Kobbie Mainoo", "Bruno Fernandes"]

N_PROFILES = 15

SCOUT_WATCH = [
    "Boubacar Kamara",
    "Ryan Gravenberch",
    "Carlos Baleba",
    "Lucien Agoumé",
    "Amadou Onana",
]

# Players to show in the "correctly excluded" section.
# Negative controls (must fail) and notable eye-test names that fell out.
EXCLUDED_PLAYERS = [
    "Bruno Fernandes",
    "Jude Bellingham",
    "Florian Wirtz",
    "Elliot Anderson",
    "Mateus Fernandes",   # spelled "Mateus" in Sofascore data
]

# 8 metrics for the percentile bar display (DataMB-style)
PROFILE_METRICS = [
    ("recoveries",        "Ball recoveries"),
    ("tackles_won_pct",   "Tackle win %"),
    ("interceptions",     "Interceptions"),
    ("own_half_passes",   "Own-half passes"),
    ("pass_cmp_pct",      "Pass completion %"),
    ("passes_final_third","Final-third passes"),
    ("possession_lost",   "Ball security"),    # count_flip: high pct = secure
    ("key_passes",        "Key passes"),
]

FEATURE_LABELS = {
    "recoveries":         "ball recoveries",
    "tackles_won_pct":    "tackle win rate",
    "interceptions":      "interceptions",
    "own_half_passes":    "own-half passing volume",
    "pass_cmp_pct":       "pass completion",
    "passes_final_third": "final-third distribution",
    "possession_lost":    "ball security",
    "key_passes":         "chance creation",
    "dispossessed":       "press resistance",
    "aerial_duels_won":   "aerial duels won",
    "aerials_won_pct":    "aerial duel win rate",
    "clearances":         "clearances made",
    "blocks":             "shots blocked",
    "tackles":            "tackle volume",
    "passes_total":       "passing volume",
    "long_ball_cmp_pct":  "long ball accuracy",
    "xa":                 "expected assists",
    "touches":            "ball involvement",
    "shots":              "shot restraint",
    "xg":                 "goal threat restraint",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ordinal(n: int) -> str:
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


def find_player(fp: pd.DataFrame, name: str) -> pd.Series | None:
    last = name.split()[-1]
    hits = fp[fp["player"].str.contains(last, case=False, na=False)]
    if hits.empty:
        return None
    return hits.sort_values("minutes", ascending=False).iloc[0]


def contract_sentence(contract_end) -> str:
    try:
        year = int(contract_end)
    except (TypeError, ValueError):
        return "Contract status unknown."
    if year <= 2026:
        return "Expiring deal — realistic this summer at low or no cost."
    if year == 2027:
        return "Entering final year — leverage exists for a negotiated exit."
    if year == 2028:
        return "Two years remaining — a moderate fee should be sufficient."
    return f"Long contract (until {year}) — significant transfer fee required."


def gap_fill_lines(player_name: str, all_fp: pd.DataFrame,
                   z_cols: list[str], feat_names: list[str],
                   gap: np.ndarray, positive_gap: np.ndarray,
                   pair_vecs: list[np.ndarray]) -> list[str]:
    """Top 2 gap dimensions where this player's z-score covers what Mainoo+Bruno lack."""
    hit = find_player(all_fp, player_name)
    if hit is None:
        return []
    p_vec = hit[z_cols].fillna(0).to_numpy(dtype=float)
    contributions = p_vec * positive_gap   # only positive gaps count

    lines = []
    for i in np.argsort(contributions)[::-1]:
        if positive_gap[i] <= 0.05:
            continue
        if p_vec[i] <= 0:
            continue
        feat     = feat_names[i]
        pair_avg = np.mean([v[i] for v in pair_vecs])
        label    = FEATURE_LABELS.get(feat, feat)
        lines.append(
            f"Covers {label} gap "
            f"(pair avg z={pair_avg:+.2f} → candidate z={p_vec[i]:+.2f})"
        )
        if len(lines) == 2:
            break
    return lines


def weakest_display_feature(row: pd.Series) -> str:
    pcts = {feat: row.get(f"pct_{feat}", np.nan) for feat, _ in PROFILE_METRICS}
    pcts = {k: v for k, v in pcts.items() if pd.notna(v)}
    if not pcts:
        return "insufficient data to identify a modelling weakness"
    worst = min(pcts, key=lambda f: pcts[f])
    label = FEATURE_LABELS.get(worst, worst)
    return f"sits at just the {ordinal(int(pcts[worst]))} percentile for {label}"


def wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    words = text.split()
    lines, line = [], indent
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = indent + word
        else:
            line = (line + " " + word) if line.strip() else indent + word
    if line.strip():
        lines.append(line)
    return "\n".join(lines)


def build_scouting_note(row: pd.Series, gap_lines: list[str],
                         closest_seed: str) -> str:
    last = row["player"].split()[-1]

    note = (
        f"{last} profiles closest to {closest_seed} "
        f"(Stage 1 archetype rank {int(row['archetype_rank'])})."
    )

    rec_pct = row.get("pct_recoveries", np.nan)
    int_pct = row.get("pct_interceptions", np.nan)
    if pd.notna(rec_pct) and pd.notna(int_pct):
        note += (
            f" Defensively, {last} sits at the {ordinal(int(rec_pct))} percentile "
            f"for ball recoveries and {ordinal(int(int_pct))} for interceptions"
        )
        if gap_lines:
            tail = gap_lines[0].replace("Covers ", "").replace("covers ", "")
            note += f", directly covering the {tail}."
        else:
            note += "."

    if len(gap_lines) >= 2:
        tail2 = gap_lines[1][0].lower() + gap_lines[1][1:]
        note += f" Also {tail2}."

    age_str = f"age {int(row['tm_age'])}" if pd.notna(row.get("tm_age")) else "age unknown"
    val_str = f"€{row['tm_value_m']:.0f}M" if pd.notna(row.get("tm_value_m")) else "value undisclosed"
    con_str = str(int(row["tm_contract_end"])) if pd.notna(row.get("tm_contract_end")) else "unknown"
    note += (
        f" Practically: {age_str}, valued at {val_str}, contract until {con_str}. "
        + contract_sentence(row.get("tm_contract_end"))
    )

    weak = weakest_display_feature(row)
    note += f" Main modelling caveat: {last} {weak}."

    return note


def print_profile(row: pd.Series, stage3_rank: int, gap_lines: list[str],
                   closest_seed: str, note: str) -> None:
    sep = "=" * 80
    arch_rank = int(row["archetype_rank"]) if pd.notna(row.get("archetype_rank")) else "?"

    print(sep)
    print(f"#{stage3_rank} (Stage 3)  {row['player'].upper()} — {row['team']} · {row['league']}")
    print(sep)

    age_str = f"{int(row['tm_age'])}" if pd.notna(row.get("tm_age")) else "?"
    val_str = f"€{row['tm_value_m']:.0f}M" if pd.notna(row.get("tm_value_m")) else "?"
    con_str = f"{int(row['tm_contract_end'])}" if pd.notna(row.get("tm_contract_end")) else "?"
    print(
        f"Age: {age_str} ({row.get('age_label', '?')})   "
        f"Value: {val_str}   Contract: {con_str}   "
        f"Stage3 score: {row.get('stage3_score', 0):.3f}"
    )
    print()
    print(
        f"  Archetype rank: {arch_rank}    "
        f"Comp score: {row.get('complementarity_score', 0):.3f}    "
        f"Combined: {row.get('combined_score', 0):.3f}    "
        f"Closest seed: {closest_seed}"
    )
    print()
    print("  PERCENTILES  (vs main-track midfielders, ≥2000 min, top-7 leagues)")
    for feat, label in PROFILE_METRICS:
        pct = row.get(f"pct_{feat}", np.nan)
        if pd.notna(pct):
            bar     = int(round(pct / 5))
            bar_str = "█" * bar + "░" * (20 - bar)
            pct_str = ordinal(int(round(pct)))
        else:
            bar_str = "░" * 20
            pct_str = "?"
        print(f"  {label:24s}  {bar_str}  {pct_str:>5s}")

    print()
    if gap_lines:
        print("  GAP FILL (what Mainoo + Bruno lack, this player covers):")
        for gl in gap_lines:
            print(f"    · {gl}")
    else:
        print("  GAP FILL: no dominant gap dimensions covered")

    print()
    print("  SCOUTING NOTE:")
    print(wrap(note, width=76, indent="  "))
    print()


# ---------------------------------------------------------------------------
# Correctly-excluded section
# ---------------------------------------------------------------------------

# One-line role labels derived from the player's top z-score features.
# Iterated in priority order; first matching rule wins.
# Note: xg/shots are count_flip — high z = low raw value = "good" for a #6.
# To catch attacking profiles via xg we handle the negative-z case separately
# in _role_from_fingerprint (a very negative xg z = high xG output = attacker).
_ROLE_RULES: list[tuple[str, str]] = [
    ("xa",                 "chance-creator / #10"),
    ("key_passes",         "chance-creator / #10"),
    ("passes_final_third", "progressive distributor"),
    ("recoveries",         "defensive midfielder / #6"),
    ("tackles",            "defensive midfielder / #6"),
    ("interceptions",      "defensive midfielder / #6"),
]


def _role_from_fingerprint(feat_names: list[str], p_vec: np.ndarray) -> str:
    """
    Derive a one-line role label from the player's z-score profile.
    Iterates _ROLE_RULES in priority order; first rule whose feature appears
    in the player's top-4 z-scores (above threshold) wins.
    For count_flip features (shots, xg, dispossessed, possession_lost) a high z-score
    means the raw value is LOW — skip them as positive identifiers.
    """
    count_flip = {"shots", "xg", "dispossessed", "possession_lost"}
    top_idx    = set(np.argsort(p_vec)[::-1][:4].tolist())

    # Priority-ordered check. Rules are split around the xG heuristic:
    # xa and key_passes first (creation-dominant players get labelled correctly),
    # then xG attacker check (Bellingham-type: high shot output, no creation features
    # in top-4), then pass-progression and defensive rules.
    CREATION_RULES  = [r for r in _ROLE_RULES if r[0] in {"xa", "key_passes"}]
    REMAINING_RULES = [r for r in _ROLE_RULES if r[0] not in {"xa", "key_passes"}]

    for rule_feat, label in CREATION_RULES:
        if rule_feat in feat_names:
            fi = feat_names.index(rule_feat)
            if fi in top_idx and p_vec[fi] > 0.5:
                return label

    # High xG output (z_xg < -2.0 = high raw xG = shoots a lot = attacker profile)
    if "xg" in feat_names and p_vec[feat_names.index("xg")] < -2.0:
        return "goal-threat B2B / #8-10"

    for rule_feat, label in REMAINING_RULES:
        if rule_feat in feat_names:
            fi = feat_names.index(rule_feat)
            if fi in top_idx and rule_feat not in count_flip and p_vec[fi] > 0.5:
                return label

    # Fallback: describe by highest positive non-flip dimension
    top_raw = [
        (feat_names[i], p_vec[i])
        for i in sorted(top_idx, key=lambda x: p_vec[x], reverse=True)
        if feat_names[i] not in count_flip and p_vec[i] > 0
    ]
    if top_raw:
        return f"{FEATURE_LABELS.get(top_raw[0][0], top_raw[0][0])}-dominant midfielder"
    return "atypical profile"


def print_excluded_section(
    all_fp:     pd.DataFrame,
    seeds_fp:   pd.DataFrame,
    ranked_main: pd.DataFrame,
    feat_names: list[str],
    z_cols:     list[str],
    weights:    np.ndarray,
    one_sided:  np.ndarray,
) -> None:
    """
    Print archetype rank and role for each player in EXCLUDED_PLAYERS.
    Shows the filter is discriminating, not silently dropping players.
    """
    rank_map = dict(zip(ranked_main["player"], ranked_main["archetype_rank"].astype(int)))
    seed_Z   = seeds_fp[z_cols].fillna(0).values
    sep = "=" * 80

    print(f"\n{sep}")
    print("CORRECTLY EXCLUDED — negative controls and eye-test names that fell out")
    print(f"{sep}")
    print(
        "Players listed here fail the archetype filter. Their rank and the features\n"
        "driving their distance confirm the filter is working as intended.\n"
    )

    for name in EXCLUDED_PLAYERS:
        last = name.split()[-1]
        hits = all_fp[all_fp["player"].str.contains(last, case=False, na=False)]
        if hits.empty:
            print(f"  {name}: not found in dataset\n")
            continue

        # For "Fernandes", pick the right one by first-name prefix
        first = name.split()[0].lower()
        exact = hits[hits["player"].str.lower().str.startswith(first)]
        row   = (exact if not exact.empty else hits).sort_values("minutes", ascending=False).iloc[0]

        pname     = row["player"]
        arch_rank = rank_map.get(pname)
        if arch_rank is None:
            print(f"  {pname}: not in main-track ranking (below {2000}-min threshold)\n")
            continue

        p_vec = row[z_cols].fillna(0).to_numpy(dtype=float)
        role  = _role_from_fingerprint(feat_names, p_vec)

        # Distance to closest seed
        seed_dists = []
        for si, s_vec in enumerate(seed_Z):
            raw_d = p_vec - s_vec
            adj_d = raw_d.copy()
            adj_d[one_sided] = np.minimum(adj_d[one_sided], 0.0)
            d = float(np.sqrt((adj_d ** 2 * weights ** 2).sum()))
            seed_dists.append((seeds_fp.iloc[si]["player"], d, s_vec))
        seed_dists.sort(key=lambda x: x[1])
        closest_name, closest_d, closest_vec = seed_dists[0]

        # Top 3 features driving distance
        raw_d = p_vec - closest_vec
        adj_d = raw_d.copy()
        adj_d[one_sided] = np.minimum(adj_d[one_sided], 0.0)
        feat_contrib = adj_d ** 2 * weights ** 2
        top_idx = np.argsort(feat_contrib)[::-1][:3]

        print(f"  {pname} ({row['team']}) — archetype rank {arch_rank}")
        print(f"  Role fingerprint: {role}")
        driver_parts = []
        for i in top_idx:
            fn = feat_names[i]
            label = FEATURE_LABELS.get(fn, fn)
            direction = "excess" if p_vec[i] > closest_vec[i] else "deficit"
            driver_parts.append(f"{label} {direction} ({p_vec[i]:+.1f}σ vs seed {closest_vec[i]:+.1f}σ)")
        print(f"  Distance drivers: {'; '.join(driver_parts)}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Rebuild fingerprints ──────────────────────────────────────────────────
    agg       = load_and_aggregate()
    main_pool = agg[agg["minutes"] >= MIN_MINUTES_MAIN].copy().reset_index(drop=True)
    ref_copy  = main_pool.copy()
    main_fp   = build_fingerprint(main_pool.copy(), ref_pool=ref_copy)

    sub_pool = agg[
        (agg["minutes"] >= MIN_MINUTES_PROSPECT)
        & (agg["minutes"] < MIN_MINUTES_MAIN)
    ].copy().reset_index(drop=True)
    ref_copy2 = main_pool.copy()
    sub_fp    = (
        build_fingerprint(sub_pool.copy(), ref_pool=ref_copy2)
        if not sub_pool.empty else pd.DataFrame()
    )
    all_fp = pd.concat([main_fp, sub_fp], ignore_index=True) if not sub_fp.empty else main_fp

    feat_names = [f[0] for f in FEATURES if f"z_{f[0]}" in main_fp.columns]
    z_cols     = [f"z_{f}" for f in feat_names]

    # ── Percentile layer ──────────────────────────────────────────────────────
    for feat in feat_names:
        col = f"f_{feat}"
        if col in main_fp.columns:
            main_fp[f"pct_{feat}"] = main_fp[col].rank(pct=True) * 100

    print(f"Percentile layer: {len(feat_names)} features, {len(main_fp)} players in reference pool\n")

    # ── Gap vector (same logic as Stage 2) ───────────────────────────────────
    pair_vecs = []
    for name in EXISTING_PAIR:
        row = find_player(all_fp, name)
        if row is None:
            raise RuntimeError(f"Could not find {name!r} in fingerprint")
        pair_vecs.append(row[z_cols].fillna(0).to_numpy(dtype=float))

    existing_z   = np.mean(pair_vecs, axis=0)
    gap          = 0.0 - existing_z
    positive_gap = np.maximum(gap, 0.0)

    # ── Load Stage 3 + closest_seed map ──────────────────────────────────────
    stage3      = pd.read_parquet(STAGE3_PATH)
    ranked_main = pd.read_parquet(RANKED_MAIN)
    seed_map    = dict(zip(ranked_main["player"], ranked_main["closest_seed"]))

    non_seeds = stage3[~stage3["player"].isin(SEED_PLAYERS)].copy()
    top_n     = non_seeds.head(N_PROFILES).copy()

    # Append any scout watch players not already in top_n
    extra = []
    for name in SCOUT_WATCH:
        last   = name.split()[-1]
        in_top = top_n["player"].str.contains(last, case=False, na=False).any()
        if not in_top:
            hit = non_seeds[non_seeds["player"].str.contains(last, case=False, na=False)]
            if not hit.empty:
                extra.append(hit.iloc[0])

    if extra:
        top_n = pd.concat([top_n, pd.DataFrame(extra)], ignore_index=True)

    # ── Join percentile columns ───────────────────────────────────────────────
    pct_cols = [f"pct_{f}" for f in feat_names if f"pct_{f}" in main_fp.columns]
    profiles  = top_n.merge(main_fp[["player"] + pct_cols], on="player", how="left")

    # ── Generate profiles ─────────────────────────────────────────────────────
    saved_rows = []

    for _, row in profiles.iterrows():
        s3_rank      = int(row["stage3_rank"]) if pd.notna(row.get("stage3_rank")) else 999
        closest_seed = seed_map.get(row["player"], "closest seed")
        g_lines      = gap_fill_lines(
            row["player"], all_fp, z_cols, feat_names, gap, positive_gap, pair_vecs
        )
        note = build_scouting_note(row, g_lines, closest_seed)
        print_profile(row, s3_rank, g_lines, closest_seed, note)

        saved_rows.append({
            "player":               row["player"],
            "stage3_rank":          row.get("stage3_rank"),
            "stage2_rank":          row.get("stage2_rank"),
            "team":                 row.get("team"),
            "league":               row.get("league"),
            "archetype_rank":       row.get("archetype_rank"),
            "closest_seed":         closest_seed,
            "complementarity_score": row.get("complementarity_score"),
            "combined_score":       row.get("combined_score"),
            "stage3_score":         row.get("stage3_score"),
            "tm_age":               row.get("tm_age"),
            "age_label":            row.get("age_label"),
            "tm_value_m":           row.get("tm_value_m"),
            "tm_contract_end":      row.get("tm_contract_end"),
            "gap_fill_1":           g_lines[0] if len(g_lines) > 0 else None,
            "gap_fill_2":           g_lines[1] if len(g_lines) > 1 else None,
            "scouting_note":        note,
            **{f"pct_{f}": row.get(f"pct_{f}") for f in feat_names},
        })

    out_df = pd.DataFrame(saved_rows)
    out_df.to_parquet(OUT_DIR / "stage4_profiles.parquet", index=False)
    print(f"Saved: {OUT_DIR / 'stage4_profiles.parquet'}  ({len(out_df)} profiles)")

    # ── Correctly-excluded section ────────────────────────────────────────────
    from sofascore_pipeline import WEIGHTS, ONE_SIDED_FEATURES, SEED_NAMES

    seeds_fp   = main_fp[main_fp["player"].isin(SEED_NAMES)].drop_duplicates("player")
    w_arr      = np.array([WEIGHTS.get(f, 1.0) for f in feat_names])
    osi_arr    = np.array([f in ONE_SIDED_FEATURES for f in feat_names])

    print_excluded_section(
        all_fp      = all_fp,
        seeds_fp    = seeds_fp,
        ranked_main = ranked_main,
        feat_names  = feat_names,
        z_cols      = z_cols,
        weights     = w_arr,
        one_sided   = osi_arr,
    )


if __name__ == "__main__":
    main()
