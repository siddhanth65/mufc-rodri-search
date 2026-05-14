"""
data_loader.py — Full multi-league, multi-season, multi-stat-type data pull.

Uses ScraperFC which handles Cloudflare via real Chrome browser.
Caches each (league, season, stat_type) to data/raw/ as parquet.
Subsequent runs load from cache — no re-scraping.

Usage:
    python src/data_loader.py --smoke   # one league, one season, one stat_type
    python src/data_loader.py --full    # everything (~40-60 mins first run)
    python src/data_loader.py --stat passing  # one stat_type, all leagues/seasons
"""

from __future__ import annotations
import time
import argparse
from pathlib import Path

import pandas as pd
from ScraperFC import FBref

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEAGUES: list[str] = [
    "England Premier League",
    "Spain La Liga",
    "Germany Bundesliga",
    "Italy Serie A",
    "France Ligue 1",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
]

SEASONS: list[str] = ["2023-2024", "2024-2025"]

# All stat types we need for the 27-feature fingerprint
STAT_TYPES: list[str] = [
    "passing",
    "defensive",
    "possession",
    "goal and shot creation",
    "shooting",
    "misc",
]

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT = 6  # seconds between requests — do not reduce

# ---------------------------------------------------------------------------
# Column standardisation
# ---------------------------------------------------------------------------

# After flattening MultiIndex, these ugly prefixes appear for identity columns.
# Map them to clean names.
RENAME_MAP = {
    "Unnamed: 1_level_0_Player": "player",
    "Unnamed: 2_level_0_Nation": "nation",
    "Unnamed: 3_level_0_Pos": "pos",
    "Unnamed: 4_level_0_Squad": "team",
    "Unnamed: 5_level_0_Age": "age",
    "Unnamed: 6_level_0_Born": "born",
    "Unnamed: 7_level_0_90s": "90s",
    "Unnamed: 0_level_0_Rk": "rk",
}


def _clean(df: pd.DataFrame, league: str, season: str, stat_type: str) -> pd.DataFrame:
    """Flatten MultiIndex columns, standardise identity cols, attach metadata."""
    df = df.copy()

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(l) for l in col).strip("_") for col in df.columns]

    # Rename identity columns
    df = df.rename(columns=RENAME_MAP)

    # Drop repeated header rows FBref embeds mid-table
    if "player" in df.columns:
        df = df[df["player"].notna()].copy()
        df = df[df["player"] != "Player"].copy()

    # Drop rows with no player name
    if "player" in df.columns:
        df = df[df["player"].str.strip() != ""].copy()

    # Attach metadata
    df["league"] = league
    df["season"] = season
    df["stat_type"] = stat_type

    # Numeric coercion — everything except string identity cols
    str_cols = {
        "player",
        "nation",
        "pos",
        "team",
        "age",
        "born",
        "league",
        "season",
        "stat_type",
        "rk",
        "Player ID",
    }
    for col in df.columns:
        if col not in str_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single pull
# ---------------------------------------------------------------------------


def pull_one(
    scraper: FBref,
    league: str,
    season: str,
    stat_type: str,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Pull one (league, season, stat_type). Returns None on failure."""
    safe_league = league.replace(" ", "_")
    cache_path = CACHE_DIR / f"{safe_league}__{season}__{stat_type}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    print(f"  Scraping: {league} | {season} | {stat_type}")
    try:
        result = scraper.scrape_stats(
            year=season,
            league=league,
            stat_category=stat_type,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

    if "player" not in result or result["player"] is None:
        print(f"  NO PLAYER DATA returned")
        return None

    df = _clean(result["player"], league, season, stat_type)
    df.to_parquet(cache_path, index=False)
    print(f"  OK — {len(df)} rows -> {cache_path.name}")
    return df


# ---------------------------------------------------------------------------
# Batch pull
# ---------------------------------------------------------------------------


def pull_stat_type(
    stat_type: str,
    leagues: list[str] = LEAGUES,
    seasons: list[str] = SEASONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull one stat_type across all leagues and seasons. Returns combined df."""
    combined_path = CACHE_DIR / f"combined__{stat_type}.parquet"
    if combined_path.exists() and not force_refresh:
        print(f"Loading combined '{stat_type}' from cache.")
        return pd.read_parquet(combined_path)

    scraper = FBref()
    frames = []
    n = len(leagues) * len(seasons)
    i = 0

    print(f"\n{'='*60}")
    print(f"Pulling '{stat_type}' — {len(leagues)} leagues x {len(seasons)} seasons")
    print(f"{'='*60}")

    for league in leagues:
        for season in seasons:
            i += 1
            print(f"\n[{i}/{n}]", end=" ")
            df = pull_one(scraper, league, season, stat_type, force_refresh)
            if df is not None:
                frames.append(df)
            if i < n:
                print(f"  Waiting {RATE_LIMIT}s...")
                time.sleep(RATE_LIMIT)

    if not frames:
        raise RuntimeError(f"No data pulled for '{stat_type}'.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_path, index=False)
    print(
        f"\nSaved combined '{stat_type}': {len(combined)} rows -> {combined_path.name}"
    )
    return combined


def pull_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Pull all stat types. ~40-60 mins first run."""
    return {st: pull_stat_type(st, force_refresh=force_refresh) for st in STAT_TYPES}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="One league, one season, passing only"
    )
    parser.add_argument(
        "--full", action="store_true", help="All leagues, seasons, stat types"
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=None,
        help="Pull one stat_type across all leagues/seasons",
    )
    args = parser.parse_args()

    if args.smoke:
        print("SMOKE TEST — passing, Premier League 2024-25\n")
        scraper = FBref()
        df = pull_one(
            scraper,
            "England Premier League",
            "2024-2025",
            "passing",
            force_refresh=True,
        )
        if df is not None:
            print(f"\nShape: {df.shape}")
            print(f"\nClean columns:\n{df.columns.tolist()}")
            id_cols = [
                c for c in ["player", "team", "pos", "season"] if c in df.columns
            ]
            print(f"\nFirst 5 rows:\n{df[id_cols].head()}")
            expected = [
                "Total_Cmp%",
                "Short_Cmp%",
                "Medium_Cmp%",
                "Long_Cmp%",
                "PrgP",
                "KP",
                "Unnamed: 25_level_0_1/3",
                "Unnamed: 26_level_0_PPA",
            ]
            print("\nKey column check:")
            for col in expected:
                found = col in df.columns
                # also fuzzy check
                fuzzy = [c for c in df.columns if col.split("_")[-1] in c]
                print(
                    f"  {'OK' if found or fuzzy else 'MISSING':8s} {col:35s} -> {fuzzy[:2]}"
                )

    elif args.stat:
        pull_stat_type(args.stat)

    elif args.full:
        print("FULL PULL — all leagues, seasons, stat_types")
        print("Expected time: 40-60 minutes first run\n")
        pull_all()
        print("\nDone. All data cached in data/raw/")

    else:
        print("Specify --smoke, --full, or --stat <stat_type>")
        print("Example: python src/data_loader.py --smoke")
