"""
fbref_scraper.py — Selenium-based FBref scraper.

Covers the stat types that ScraperFC fails on (complex MultiIndex tables):
  passing, pass types, defensive, possession, goal and shot creation

Saves to data/raw/combined__{stat_type}.parquet — same naming that
fingerprint.py expects, overwriting the broken ScraperFC outputs.

Does NOT touch shooting or misc (ScraperFC handles those fine).

Usage:
    python src/fbref-scraper.py --smoke           # one league, one season, passing
    python src/fbref-scraper.py --full            # all leagues x seasons x stat types
    python src/fbref-scraper.py --stat passing    # one stat type, all leagues/seasons
"""

from __future__ import annotations
import time
import re
import argparse
from pathlib import Path
from io import StringIO

import requests as _req
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

try:
    import undetected_chromedriver as uc
    _UC_AVAILABLE = True
except ImportError:
    _UC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEAGUES: dict[str, tuple[str, str]] = {
    "ENG-Premier League": ("9", "Premier-League"),
    "ESP-La Liga": ("12", "La-Liga"),
    "GER-Bundesliga": ("20", "Bundesliga"),
    "ITA-Serie A": ("11", "Serie-A"),
    "FRA-Ligue 1": ("13", "Ligue-1"),
    "NED-Eredivisie": ("23", "Eredivisie"),
    "POR-Primeira Liga": ("32", "Primeira-Liga"),
}

SEASONS: list[str] = ["2023-2024", "2024-2025"]

# logical_name (used in cache filename & fingerprint.py) -> FBref URL slug
STAT_TYPES: dict[str, str] = {
    "passing": "passing",
    "pass types": "passing_types",
    "defensive": "defense",
    "possession": "possession",
    "goal and shot creation": "gca",
}

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_SECONDS = 6  # seconds between requests

# ---------------------------------------------------------------------------
# Column renaming — FBref MultiIndex flattened names -> clean logical names
# ---------------------------------------------------------------------------

# These are the Unnamed top-level headers that FBref uses for identity columns.
# The exact Unnamed index can vary slightly across stat types; we handle all variants.
RENAME_MAP = {
    "Unnamed: 0_level_0_Rk": "rk",
    "Unnamed: 1_level_0_Player": "player",
    "Unnamed: 2_level_0_Nation": "nation",
    "Unnamed: 3_level_0_Pos": "pos",
    "Unnamed: 4_level_0_Squad": "team",
    "Unnamed: 5_level_0_Age": "age",
    "Unnamed: 6_level_0_Born": "born",
    "Unnamed: 7_level_0_90s": "90s",
    # flat-header variants (non-MultiIndex pages or already-rendered tables)
    "Rk": "rk",
    "Player": "player",
    "Nation": "nation",
    "Pos": "pos",
    "Squad": "team",
    "Age": "age",
    "Born": "born",
}

# ---------------------------------------------------------------------------
# Selenium driver
# ---------------------------------------------------------------------------


def _make_driver() -> webdriver.Chrome:
    if _UC_AVAILABLE:
        opts = uc.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        return uc.Chrome(options=opts)

    # Fallback: plain selenium — less reliable against Cloudflare
    print("WARNING: undetected_chromedriver not installed. Install with:")
    print("  pip install undetected-chromedriver")
    print("Falling back to plain selenium (likely blocked by Cloudflare)\n")
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ---------------------------------------------------------------------------
# Raw HTML fetcher — bypasses JS rendering to get comment-embedded tables
# ---------------------------------------------------------------------------


def _fetch_raw_html(driver: webdriver.Chrome, url: str) -> str | None:
    """
    Fetch the page's raw server HTML via requests, using CF clearance cookies
    from the UC browser session. The raw HTML still has FBref's tables in
    HTML comments (before JS uncomments them), which always contain full data.
    Returns None if the request fails or returns short/error content.
    """
    try:
        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
        ua = driver.execute_script("return navigator.userAgent;")
        resp = _req.get(
            url,
            cookies=cookies,
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://fbref.com/",
            },
            timeout=30,
        )
        print(f"    raw-HTML fetch: HTTP {resp.status_code}, {len(resp.text):,} chars")
        if resp.ok and len(resp.text) > 10_000:
            return resp.text
    except Exception as e:
        print(f"    raw-HTML fetch failed: {e}")
    return None


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def _build_url(league_id: str, league_slug: str, season: str, stat_slug: str) -> str:
    return (
        f"https://fbref.com/en/comps/{league_id}/{season}/{stat_slug}/"
        f"{season}-{league_slug}-Stats"
    )


# ---------------------------------------------------------------------------
# Page parser
# ---------------------------------------------------------------------------


def _parse_page(
    html: str, stat_slug: str, league: str, season: str
) -> pd.DataFrame | None:
    table_id = f"stats_{stat_slug}"

    # Try HTML comments FIRST — the raw source embeds tables in comments before JS
    # uncomments them. The comment version always has full data; the JS-rendered DOM
    # often has empty cells because the browser replaces the comment with a skeleton.
    tables = []
    for comment in re.findall(r"<!--(.*?)-->", html, re.DOTALL):
        if table_id in comment:
            try:
                tables = pd.read_html(StringIO(comment), attrs={"id": table_id})
                if tables:
                    break
            except Exception:
                continue

    # Fallback: main DOM (used when JS has already rendered the table into the page)
    if not tables:
        try:
            tables = pd.read_html(StringIO(html), attrs={"id": table_id})
        except Exception:
            pass

    if not tables:
        print(f"    TABLE NOT FOUND — '{table_id}' not on page")
        return None

    df = tables[0].copy()

    # Flatten MultiIndex columns (FBref uses 2-level headers for grouped stats)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(lvl) for lvl in col).strip("_") for col in df.columns
        ]

    # Rename identity columns to canonical names
    df = df.rename(columns=RENAME_MAP)

    # Drop FBref's repeated header rows (embedded every ~25 data rows)
    if "player" in df.columns:
        df = df[df["player"].notna()].copy()
        df = df[df["player"].astype(str).str.strip() != ""].copy()
        df = df[df["player"] != "Player"].copy()

    if "rk" in df.columns:
        df = df[df["rk"].astype(str) != "Rk"].copy()

    if len(df) == 0:
        print(f"    0 rows after header cleanup")
        return None

    # Diagnostic: show raw values for key columns before coercion
    _diag = ["Total_Att", "Total_Cmp%", "Unnamed: 22_level_0_Ast", "Unnamed: 24_level_0_KP"]
    for _dc in _diag:
        if _dc in df.columns:
            _vals = df[_dc].head(3).tolist()
            print(f"    PRE-COERCE {_dc} (dtype={df[_dc].dtype}): {_vals}")

    # Numeric coercion — strip commas (FBref formats "1,452") then coerce
    str_cols = {
        "player", "nation", "pos", "team", "age", "born",
        "rk", "league", "season", "stat_type", "Player ID",
    }
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

    # Attach metadata
    df["league"] = league
    df["season"] = season

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single-page scrape with retry
# ---------------------------------------------------------------------------


def _scrape_page(
    driver: webdriver.Chrome,
    url: str,
    stat_slug: str,
    league: str,
    season: str,
    max_retries: int = 2,
) -> pd.DataFrame | None:
    for attempt in range(1, max_retries + 1):
        try:
            driver.get(url)
            time.sleep(7)  # let JS render + CF auto-resolve
        except Exception as e:
            print(f"    BROWSER ERROR (attempt {attempt}): {e}")
            continue

        page_source = driver.page_source

        if "Just a moment" in page_source or "Enable JavaScript" in page_source:
            print(f"    Cloudflare challenge — waiting 25s (attempt {attempt})")
            time.sleep(25)
            page_source = driver.page_source

        # Fetch raw pre-JS HTML via requests using the CF clearance cookies UC obtained.
        # FBref's JS replaces comment-embedded tables with empty DOM skeletons before
        # populating cells, so driver.page_source has empty stat columns. The raw HTML
        # still has the complete tables inside <!-- --> comments.
        raw_html = _fetch_raw_html(driver, url)
        if raw_html:
            print(f"    using raw HTML ({len(raw_html):,} chars)")
        else:
            print(f"    raw HTML unavailable — using Selenium page_source ({len(page_source):,} chars)")
        df = _parse_page(raw_html if raw_html else page_source, stat_slug, league, season)
        if df is not None:
            return df

        if attempt < max_retries:
            print(f"    Retrying in 10s...")
            time.sleep(10)

    return None


# ---------------------------------------------------------------------------
# Batch scrape for one stat type
# ---------------------------------------------------------------------------


def scrape_stat_type(
    logical_name: str,
    leagues: dict[str, tuple[str, str]] = LEAGUES,
    seasons: list[str] = SEASONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if logical_name not in STAT_TYPES:
        raise ValueError(
            f"Unknown stat type '{logical_name}'. "
            f"Choose from: {list(STAT_TYPES.keys())}"
        )

    stat_slug = STAT_TYPES[logical_name]
    cache_path = CACHE_DIR / f"combined__{logical_name}.parquet"

    if cache_path.exists() and not force_refresh:
        print(f"  Loading '{logical_name}' from cache.")
        return pd.read_parquet(cache_path)

    n_total = len(leagues) * len(seasons)
    i = 0
    frames: list[pd.DataFrame] = []

    print(f"\n{'='*60}")
    print(f"Scraping '{logical_name}' (slug='{stat_slug}')")
    print(f"  {len(leagues)} leagues x {len(seasons)} seasons = {n_total} pages")
    print(f"{'='*60}")

    driver = _make_driver()
    try:
        for league_name, (league_id, league_slug) in leagues.items():
            for season in seasons:
                i += 1
                url = _build_url(league_id, league_slug, season, stat_slug)
                print(f"\n[{i}/{n_total}] {league_name} | {season}")
                print(f"  {url}")

                df = _scrape_page(driver, url, stat_slug, league_name, season)

                if df is not None:
                    print(f"  OK — {len(df)} rows, {len(df.columns)} cols")
                    # Spot-check: print a data column to verify it's non-null
                    data_cols = [
                        c for c in df.columns
                        if c not in ("player", "team", "pos", "nation", "age",
                                     "born", "rk", "league", "season", "90s")
                        and df[c].notna().sum() > 0
                    ]
                    if data_cols:
                        sample_col = data_cols[0]
                        nn = df[sample_col].notna().sum()
                        print(f"  Data check: {sample_col} has {nn}/{len(df)} non-null")
                    else:
                        print("  WARNING: all stat columns are NaN — table may not have rendered")
                    frames.append(df)
                else:
                    print(f"  FAILED — no data for this page")

                if i < n_total:
                    print(f"  Waiting {RATE_LIMIT_SECONDS}s...")
                    time.sleep(RATE_LIMIT_SECONDS)
    finally:
        driver.quit()

    if not frames:
        raise RuntimeError(
            f"No data scraped for '{logical_name}'. "
            f"Check if FBref is blocking requests or the table IDs have changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(cache_path, index=False)
    print(
        f"\nSaved '{logical_name}': {len(combined)} rows -> {cache_path.name}"
    )
    return combined


# ---------------------------------------------------------------------------
# Scrape all broken stat types
# ---------------------------------------------------------------------------


def scrape_all(force_refresh: bool = True) -> dict[str, pd.DataFrame]:
    return {
        name: scrape_stat_type(name, force_refresh=force_refresh)
        for name in STAT_TYPES
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One league (PL), one season (2024-25), passing only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All leagues x seasons x stat types (~25-35 mins)",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default=None,
        help=f"One stat type across all leagues/seasons. Options: {list(STAT_TYPES.keys())}",
    )
    args = parser.parse_args()

    if args.smoke:
        print("SMOKE TEST — passing, Premier League 2024-25\n")
        df = scrape_stat_type(
            "passing",
            leagues={"ENG-Premier League": ("9", "Premier-League")},
            seasons=["2024-2025"],
            force_refresh=True,
        )
        print(f"\nShape: {df.shape}")
        print(f"\nAll columns:")
        for c in sorted(df.columns):
            nn = df[c].notna().sum()
            print(f"  {c:45s} non_null={nn}")
        key_cols = ["Total_Att", "Total_Cmp%", "Short_Cmp%", "PrgP", "KP"]
        print("\nKey column check:")
        for col in key_cols:
            # also fuzzy match
            exact = col in df.columns
            fuzzy = [c for c in df.columns if col.split("_")[-1] in c.split("_")]
            status = "OK" if exact or fuzzy else "MISSING"
            print(f"  {status:8s} {col:20s} -> {fuzzy[:3]}")

    elif args.stat:
        if args.stat not in STAT_TYPES:
            print(f"Unknown stat: '{args.stat}'. Options: {list(STAT_TYPES.keys())}")
        else:
            scrape_stat_type(args.stat, force_refresh=True)

    elif args.full:
        print("FULL PULL — all 5 stat types, 7 leagues, 2 seasons (~25-35 mins)\n")
        scrape_all(force_refresh=True)
        print("\nDone. Run: python src/fingerprint.py --inspect")

    else:
        print("Specify --smoke, --full, or --stat <name>")
        print(f"Stat types: {list(STAT_TYPES.keys())}")
        print("\nExample: python src/fbref-scraper.py --smoke")
        print("Example: python src/fbref-scraper.py --stat passing")
        print("Example: python src/fbref-scraper.py --full")
