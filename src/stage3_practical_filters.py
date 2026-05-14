"""
stage3_practical_filters.py — Stage 3: Practical filters.

Enriches Stage 2 rankings with Transfermarkt data (age, market value,
contract expiry) and applies soft scoring modifiers.

Steps:
  1. Load Stage 2 ranked parquet
  2. Build player-URL map from Transfermarkt (get_player_links per league)
  3. Match Stage 2 players to TM URLs by normalised name slug
  4. Scrape individual player pages for matched players only (~80 requests)
  5. Cache results to data/processed/tm_cache.parquet
  6. Compute modifiers:
       age_mod      — prime 24-29 = 1.0, young <24 = 0.90, veteran 30+ = 0.80
       contract_mod — expiry ≤2026 = 1.10, 2027 = 1.00, 2028 = 0.92, 2029+ = 0.85
  7. stage3_score = combined_score × age_mod × contract_mod
  8. Print top 20 + scout watch

Usage:
    python src/stage3_practical_filters.py
    python src/stage3_practical_filters.py --refresh   # re-scrape even if cache exists
"""

from __future__ import annotations
import sys
import re
import unicodedata
import time
import argparse
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sofascore_pipeline import LEAGUES  # reuse league list

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR      = Path("data/processed")
STAGE2_PATH  = OUT_DIR / "stage2_ranked.parquet"
TM_CACHE     = OUT_DIR / "tm_cache.parquet"
TM_SEASON    = "24/25"

# Leagues Transfermarkt accepts — must match ScraperFC names
TM_LEAGUES = [
    "England Premier League",
    "Spain La Liga",
    "Germany Bundesliga",
    "Italy Serie A",
    "France Ligue 1",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
]

SCOUT_WATCH = [
    "Carlos Baleba",
    "Boubacar Kamara",
    "Ryan Gravenberch",
    "Lucien Agoumé",
    "Amadou Onana",
]

# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Normalise a player name to the hyphenated ASCII slug used in TM URLs."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = ascii_str.lower()
    ascii_str = re.sub(r"[^a-z0-9\s-]", "", ascii_str)
    ascii_str = re.sub(r"\s+", "-", ascii_str.strip())
    return ascii_str


def slug_from_url(url: str) -> tuple[str, str]:
    """Return (slug, player_id) from a transfermarkt URL."""
    parts = urlparse(url).path.strip("/").split("/")
    slug = parts[0] if parts else ""
    pid  = parts[-1] if parts else ""
    return slug, pid


def profil_url(slug: str, pid: str) -> str:
    return f"https://www.transfermarkt.us/{slug}/profil/spieler/{pid}"


# ---------------------------------------------------------------------------
# Transfermarkt helpers
# ---------------------------------------------------------------------------

def build_url_map(refresh: bool = False) -> dict[str, str]:
    """
    Fetch player links for each TM league and return slug → profil_url dict.
    Cached to TM_CACHE parent dir as tm_url_map.parquet.
    """
    map_cache = OUT_DIR / "tm_url_map.parquet"
    if map_cache.exists() and not refresh:
        df = pd.read_parquet(map_cache)
        print(f"Loaded URL map from cache ({len(df)} entries)")
        return dict(zip(df["slug"], df["url"]))

    from ScraperFC import Transfermarkt
    tm = Transfermarkt()

    records = []
    for league in TM_LEAGUES:
        print(f"  Fetching player links: {league} …")
        try:
            links = tm.get_player_links(TM_SEASON, league)
            for url in links:
                s, pid = slug_from_url(url)
                if s and pid.isdigit():
                    records.append({"slug": s, "pid": pid, "url": profil_url(s, pid)})
        except Exception as exc:
            print(f"    WARNING: {exc}")

    df = pd.DataFrame(records).drop_duplicates("slug")
    df.to_parquet(map_cache, index=False)
    print(f"URL map built: {len(df)} unique players across {len(TM_LEAGUES)} leagues")
    return dict(zip(df["slug"], df["url"]))


def match_players(names: list[str], url_map: dict[str, str]) -> dict[str, str]:
    """Return {player_name: tm_url} for names we can match."""
    import difflib
    slugs = list(url_map.keys())
    matched: dict[str, str] = {}

    for name in names:
        s = slugify(name)
        if s in url_map:
            matched[name] = url_map[s]
            continue
        # Try just the last name
        last = slugify(name.split()[-1])
        candidates = [sl for sl in slugs if sl.endswith(last) or sl.startswith(last)]
        if candidates:
            matched[name] = url_map[candidates[0]]
            continue
        # Fuzzy
        close = difflib.get_close_matches(s, slugs, n=1, cutoff=0.75)
        if close:
            matched[name] = url_map[close[0]]

    return matched


def parse_value(raw: str | None) -> float | None:
    """Parse TM market value string (e.g. '€45.00m', '€500Th.') to float millions."""
    if not raw:
        return None
    raw = str(raw).strip()
    m = re.search(r"[\d,\.]+", raw)
    if not m:
        return None
    num = float(m.group().replace(",", ""))
    if "bn" in raw.lower():
        return num * 1000
    if "th" in raw.lower() or "k" in raw.lower():
        return num / 1000
    return num  # already in millions


def parse_contract_year(raw: str | None) -> int | None:
    """Extract 4-digit year from contract expiry string."""
    if not raw:
        return None
    m = re.search(r"\b(20\d{2})\b", str(raw))
    return int(m.group(1)) if m else None


def scrape_players(names: list[str], url_map: dict[str, str],
                   refresh: bool = False) -> pd.DataFrame:
    """Scrape TM for each player in names; merge with cache."""
    if TM_CACHE.exists() and not refresh:
        cached = pd.read_parquet(TM_CACHE)
        already = set(cached["player"].tolist())
        todo = [n for n in names if n not in already]
        if not todo:
            print(f"All players found in TM cache ({len(cached)} entries)")
            return cached
        print(f"Cache hit for {len(already)} players; scraping {len(todo)} new …")
    else:
        cached = pd.DataFrame()
        todo = names

    matched = match_players(todo, url_map)
    print(f"  Matched {len(matched)}/{len(todo)} players to TM URLs")

    from ScraperFC import Transfermarkt
    tm = Transfermarkt()

    rows = []
    for name in todo:
        url = matched.get(name)
        if not url:
            print(f"  ✗ No URL match: {name}")
            rows.append({"player": name, "tm_age": None, "tm_value_m": None,
                         "tm_contract_end": None, "tm_url": None})
            continue
        try:
            df = tm.scrape_player(url)
            row = df.iloc[0]
            age_raw  = row.get("Age")
            val_raw  = row.get("Value")
            con_raw  = row.get("Contract expiration")
            rows.append({
                "player":          name,
                "tm_age":          int(age_raw) if age_raw is not None else None,
                "tm_value_m":      parse_value(val_raw),
                "tm_contract_end": parse_contract_year(con_raw),
                "tm_url":          url,
            })
            print(f"  ✓ {name:30s} age={age_raw}  val={val_raw}  contract={con_raw}")
            time.sleep(0.4)  # polite crawl
        except Exception as exc:
            print(f"  ✗ Scrape failed for {name}: {exc}")
            rows.append({"player": name, "tm_age": None, "tm_value_m": None,
                         "tm_contract_end": None, "tm_url": None})

    new_data = pd.DataFrame(rows)
    combined = pd.concat([cached, new_data], ignore_index=True) if not cached.empty else new_data
    combined.to_parquet(TM_CACHE, index=False)
    return combined


# ---------------------------------------------------------------------------
# Scoring modifiers
# ---------------------------------------------------------------------------

def age_modifier(age) -> float:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return 1.0  # unknown → neutral
    if 24 <= a <= 29:
        return 1.00  # prime
    if 22 <= a < 24:
        return 0.92  # project — one step away from prime
    if a < 22:
        return 0.85  # long project
    if 30 <= a <= 31:
        return 0.88  # declining but usable
    return 0.78      # 32+


def contract_modifier(contract_end) -> float:
    try:
        year = int(contract_end)
    except (TypeError, ValueError):
        return 1.0  # unknown → neutral
    if year <= 2026:
        return 1.10  # expiring — cheap to sign or free
    if year == 2027:
        return 1.00
    if year == 2028:
        return 0.92
    return 0.85      # 2029+


def age_label(age) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if a < 23:
        return "prospect"
    if a <= 29:
        return "prime"
    return "veteran"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(refresh: bool = False) -> None:
    # ── Load Stage 2 ─────────────────────────────────────────────────────────
    stage2 = pd.read_parquet(STAGE2_PATH)
    players = stage2["player"].tolist()
    print(f"Stage 2 pool: {len(players)} players")

    # ── Build Transfermarkt URL map ───────────────────────────────────────────
    print("\nBuilding TM URL map …")
    url_map = build_url_map(refresh=refresh)

    # ── Scrape TM for each player ─────────────────────────────────────────────
    print("\nScraping Transfermarkt player pages …")
    tm_data = scrape_players(players, url_map, refresh=refresh)

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = stage2.merge(tm_data[["player", "tm_age", "tm_value_m",
                                "tm_contract_end", "tm_url"]],
                      on="player", how="left")

    # ── Scoring modifiers ─────────────────────────────────────────────────────
    df["age_mod"]      = df["tm_age"].apply(age_modifier)
    df["contract_mod"] = df["tm_contract_end"].apply(contract_modifier)
    df["stage3_score"] = df["combined_score"] * df["age_mod"] * df["contract_mod"]
    df["age_label"]    = df["tm_age"].apply(age_label)

    df = df.sort_values("stage3_score", ascending=False).reset_index(drop=True)
    df["stage3_rank"] = df.index + 1

    # ── Display ───────────────────────────────────────────────────────────────
    display_cols = [
        "player", "team", "league",
        "archetype_rank", "complementarity_score", "combined_score",
        "tm_age", "age_label", "tm_value_m", "tm_contract_end",
        "age_mod", "contract_mod", "stage3_score",
    ]

    print("\n" + "=" * 80)
    print("STAGE 3 — TOP 20  (Stage 2 combined × age × contract)")
    print("=" * 80)
    pd.set_option("display.max_colwidth", 22)
    print(
        df[display_cols]
        .head(20)
        .to_string(index=False, float_format="{:.3f}".format)
    )

    # Save
    save_cols = display_cols + ["stage3_rank", "stage2_rank", "league_tier", "tm_url"]
    df[save_cols].to_parquet(OUT_DIR / "stage3_ranked.parquet", index=False)
    print(f"\nSaved: {OUT_DIR / 'stage3_ranked.parquet'}")

    # ── Scout watch ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SCOUT WATCH — Stage 3 positions")
    print("=" * 80)
    for name in SCOUT_WATCH:
        last = name.split()[-1]
        hit = df[df["player"].str.contains(last, case=False, na=False)]
        if hit.empty:
            print(f"  {'—':>4s}  {name}")
        else:
            r = hit.iloc[0]
            age_str = f"{int(r['tm_age'])}" if pd.notna(r.get("tm_age")) else "?"
            con_str = f"{int(r['tm_contract_end'])}" if pd.notna(r.get("tm_contract_end")) else "?"
            val_str = f"€{r['tm_value_m']:.1f}M" if pd.notna(r.get("tm_value_m")) else "?"
            print(
                f"  #{int(r['stage3_rank']):3d}  {r['player']:30s}"
                f"  s2={int(r['stage2_rank']):3d}"
                f"  arch={int(r['archetype_rank']):3d}"
                f"  age={age_str} ({r['age_label']})"
                f"  val={val_str}"
                f"  contract={con_str}"
                f"  score={r['stage3_score']:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Re-scrape TM even if cache exists")
    args = parser.parse_args()
    main(refresh=args.refresh)
