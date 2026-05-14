"""
Column resolver for FBref data via soccerdata.

FBref returns MultiIndex columns that we flatten with underscore-join
("Performance_npxG", "Per 90 Minutes_npxG", etc.). Column names vary across
stat_types and occasionally across seasons. This module maps logical feature
names to actual column names by checking a list of candidates per feature
and failing LOUD if none match.

No silent fallbacks. If a feature can't be resolved, we raise.
"""

from __future__ import annotations
import pandas as pd

# Map: logical name -> list of candidate column names to try, in order.
# Add candidates as we discover variants.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    # --- identifiers ---
    "player": ["player"],
    "team": ["team"],
    "pos": ["pos"],
    "minutes": ["Playing Time_Min", "Playing_Time_Min", "Min"],
    "nineties": ["Playing Time_90s", "Playing_Time_90s", "90s"],
    # --- standard ---
    "goals": ["Performance_Gls", "Gls"],
    "shots": ["Performance_Sh", "Standard_Sh", "Per 90 Minutes_Sh", "Sh"],
    "xg": ["Expected_xG", "xG"],
    "npxg": ["Per 90 Minutes_npxG", "Expected_npxG", "npxG"],
    "xa": ["Per 90 Minutes_xAG", "Expected_xAG", "xAG", "Expected_xA"],
    # --- shooting ---
    "shots": ["Performance_Sh", "Standard_Sh", "Sh"],
    "npxg": ["Expected_npxG", "Per 90 Minutes_npxG", "npxG"],
    "xa": ["Per 90 Minutes_xAG", "Expected_xAG", "xAG"],
    # --- passing ---
    "passes_total_att": ["Total_Att", "Att"],
    "pass_cmp_pct": ["Total_Cmp%", "Cmp%"],
    "short_cmp_pct": ["Short_Cmp%"],
    "medium_cmp_pct": ["Medium_Cmp%"],
    "long_cmp_pct": ["Long_Cmp%"],
    "prog_passes": ["PrgP"],
    "key_passes": ["KP"],
    "passes_final_third": ["1/3"],
    "passes_pen_area": ["PPA"],
    # --- passing_types ---
    "switches": ["Pass Types_Sw", "Sw"],
    # --- defense ---
    "tackles": ["Tackles_Tkl", "Tkl"],
    "tackles_won": ["Tackles_TklW", "TklW"],
    "interceptions": ["Int"],
    "blocks": ["Blocks_Blocks", "Blocks"],
    # --- possession ---
    "touches": ["Touches_Touches"],
    "touches_def_3rd": ["Touches_Def 3rd"],
    "touches_mid_3rd": ["Touches_Mid 3rd"],
    "touches_att_3rd": ["Touches_Att 3rd"],
    "prog_carries": ["Carries_PrgC"],
    "miscontrols": ["Miscontrols_Mis", "Mis"],
    "dispossessed": ["Miscontrols_Dis", "Dis"],
    # --- gca ---
    "sca": ["SCA_SCA"],
    "gca": ["GCA_GCA"],
    # --- misc ---
    "recoveries": ["Performance_Recov", "Recov"],
    "aerials_won_pct": ["Aerial Duels_Won%", "Aerial_Duels_Won%"],
}


class ColumnNotFoundError(KeyError):
    """Raised when no candidate matches an available column. Fail loud."""


def resolve(df: pd.DataFrame, logical_name: str) -> str:
    """Return the actual df column matching logical_name, else raise."""
    if logical_name not in COLUMN_CANDIDATES:
        raise ColumnNotFoundError(
            f"No candidate list defined for logical name '{logical_name}'. "
            f"Add it to COLUMN_CANDIDATES in src/columns.py."
        )
    for candidate in COLUMN_CANDIDATES[logical_name]:
        if candidate in df.columns:
            return candidate
    raise ColumnNotFoundError(
        f"Could not resolve '{logical_name}'. "
        f"Tried: {COLUMN_CANDIDATES[logical_name]}. "
        f"Available columns: {sorted(df.columns.tolist())[:20]}... "
        f"(showing first 20 of {len(df.columns)})"
    )


def resolve_many(df: pd.DataFrame, logical_names: list[str]) -> dict[str, str]:
    """Resolve a batch. Returns {logical: actual}. Raises on first miss."""
    return {name: resolve(df, name) for name in logical_names}


def safe_resolve(df: pd.DataFrame, logical_name: str) -> str | None:
    """Non-raising variant — returns None if no candidate matches.

    Use only when a feature is genuinely optional. Default is to raise.
    """
    try:
        return resolve(df, logical_name)
    except ColumnNotFoundError:
        return None


if __name__ == "__main__":
    """
    Smoke test — runs resolver against each stat_type separately.
    Each feature lives in exactly one stat_type; testing them together
    against 'standard' will always fail for passing/defense/etc. features.
    """
    import soccerdata as sd

    STAT_TYPE_FEATURES = {
        "standard": ["player", "team", "pos", "minutes", "goals"],
        "shooting": ["shots", "npxg"],
        "passing": ["passes_total_att", "pass_cmp_pct", "short_cmp_pct",
                    "medium_cmp_pct", "long_cmp_pct", "prog_passes",
                    "key_passes", "passes_final_third", "passes_pen_area", "xa"],
        "passing_types": ["switches"],
        "defense": ["tackles", "tackles_won", "interceptions", "blocks"],
        "possession": ["touches", "touches_def_3rd", "touches_mid_3rd",
                       "touches_att_3rd", "prog_carries",
                       "miscontrols", "dispossessed"],
        "gca": ["sca"],
        "misc": ["recoveries", "aerials_won_pct"],
    }

    all_passed = True
    for stat_type, features in STAT_TYPE_FEATURES.items():
        print(f"\nTesting resolver against stat_type='{stat_type}'...")
        fbref = sd.FBref(leagues="ENG-Premier League", seasons="2025")
        df = fbref.read_player_season_stats(stat_type=stat_type)
        df.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in df.columns
        ]
        df = df.reset_index()

        for name in features:
            try:
                actual = resolve(df, name)
                print(f"  OK    {name:30s} -> {actual}")
            except ColumnNotFoundError as e:
                print(f"  FAIL  {name:30s} -> {e}")
                all_passed = False

    print("\n" + ("ALL PASSED" if all_passed else "SOME FAILED — fix before proceeding"))
