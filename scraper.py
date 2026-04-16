"""
NCAA Men's Basketball Scraper — Sports-Reference.com
=====================================================
Scrapes regular-season game results (2008-09 through 2025-26) by:
  1. Visiting each season summary page to discover all conferences
  2. Visiting each conference's schedule page to parse games
  3. Writing one CSV per season (e.g. 2009.csv, 2010.csv, ..., 2026.csv)

Output columns:
    season, conference, date, home_team, home_rank,
    away_team, away_rank, home_score, away_score, neutral

Rate limit: Sports-Reference enforces ~10 requests/min.
This script sleeps 7 seconds between requests to stay safe.

Usage:
    pip install requests beautifulsoup4
    python scrape_cbb.py

Notes:
  - Sports-Reference uses "end year" for seasons (2008-09 => 2009).
  - Schedule tables show the home team on the RIGHT column in most
    conference schedules, with an @ symbol indicating the away team.
  - Ranks appear as "(#N)" next to team names when the team was ranked.
  - A 'N' in the game notes column indicates a neutral-site game.
"""

import csv
import os
import re
import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────

START_YEAR  = 2009   # 2008-09 season
END_YEAR    = 2026   # 2025-26 season
OUTPUT_DIR  = Path("cbb_data")
REQUEST_DELAY = 7    # seconds between requests (10 req/min = 6s; 7s is safe)
BASE_URL    = "https://www.sports-reference.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP helper ───────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-scraper/1.0; "
        "respectful of robots.txt)"
    )
})

def get(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    log.info("GET %s", url)
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 429:
            log.warning("  429 Too Many Requests — sleeping 60 s then retrying.")
            time.sleep(60)
            resp = SESSION.get(url, timeout=30)
        if resp.status_code != 200:
            log.warning("  HTTP %d – skipping.", resp.status_code)
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as exc:
        log.warning("  Request error: %s", exc)
        return None
    finally:
        time.sleep(REQUEST_DELAY)


# ── Parsing helpers ───────────────────────────────────────────────────────────

RANK_RE = re.compile(r"\((\d+)\)")

def _split_rank(raw: str) -> tuple[str, str]:
    """
    Given a raw team string like '(3) Duke' or 'Duke', return
    (team_name, rank_str) where rank_str is '' if unranked.
    """
    raw = raw.strip()
    m = RANK_RE.search(raw)
    rank = m.group(1) if m else ""
    name = RANK_RE.sub("", raw).strip()
    return name, rank


def get_conferences_for_season(year: int) -> list[tuple[str, str]]:
    """
    Scrape the season summary page and return a list of
    (conference_name, schedule_url) for every men's conference.
    
    URL pattern: /cbb/seasons/men/{year}.html
    """
    url = f"{BASE_URL}/cbb/seasons/men/{year}.html"
    soup = get(url)
    if soup is None:
        return []

    conferences = []

    # The conference links follow the pattern /cbb/conferences/{slug}/men/{year}.html
    conf_re = re.compile(r"/cbb/conferences/([^/]+)/men/\d+\.html")
    seen = set()
    for a in soup.find_all("a", href=conf_re):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        slug_match = conf_re.search(href)
        slug = slug_match.group(1)
        # Build the schedule URL for this conference + year
        schedule_url = f"{BASE_URL}/cbb/conferences/{slug}/men/{year}-schedule.html"
        conferences.append((a.get_text(strip=True), schedule_url))

    log.info("  Found %d conferences for %d.", len(conferences), year)
    return conferences


def parse_schedule_page(conf_name: str, year: int, url: str) -> list[dict]:
    """
    Parse a conference schedule page and return a list of game dicts.

    Sports-Reference schedule tables have columns roughly like:
        Date | Time | Home | (away indicated by @ prefix or separate col) | Score ...

    The actual table id is typically "schedule".
    Columns vary slightly; we detect them by header text.
    """
    soup = get(url)
    if soup is None:
        return []

    # Some pages store the table in a comment (JS-rendered); try to find it.
    table = soup.find("table", {"id": "schedule"})
    if table is None:
        # Try extracting from comments (sports-reference sometimes hides tables)
        import re as _re
        comments = soup.find_all(string=lambda t: isinstance(t, str) and "schedule" in t)
        for c in comments:
            if '<table' in c:
                inner = BeautifulSoup(c, "html.parser")
                table = inner.find("table", {"id": "schedule"})
                if table:
                    break

    if table is None:
        log.warning("  No schedule table found at %s", url)
        return []

    # Parse header to map column names → indices
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            headers.append(th.get("data-stat", th.get_text(strip=True).lower()))

    rows = []
    tbody = table.find("tbody")
    if tbody is None:
        return []

    for tr in tbody.find_all("tr"):
        # Skip spacer / section-header rows
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        # Build a dict keyed by data-stat attribute
        row_data = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            row_data[stat] = cell.get_text(strip=True)

        # Skip rows without a date (separator rows)
        date_val = row_data.get("date_game") or row_data.get("date") or ""
        if not date_val or date_val.lower() in ("date", ""):
            continue

        # Skip future/unplayed games (no score)
        # score_val = (
        #     row_data.get("score")
        #     or row_data.get("pts")
        #     or row_data.get("home_pts")
        #     or ""
        # )

        # Determine teams. Sports-Reference conference schedule pages use:
        #   visitor_school_name  (away team)
        #   home_school_name     (home team)
        # or sometimes "road_team" / "home_team"
        # away_raw = (
        #     row_data.get("visitor_school_name")
        #     or row_data.get("road_team")
        #     or row_data.get("away_team")
        #     or ""
        # )
        # home_raw = (
        #     row_data.get("home_school_name")
        #     or row_data.get("home_team")
        #     or ""
        # )

        # if not away_raw and not home_raw:
        #     continue

        away_team = row_data.get("visitor_school_name")
        home_team = row_data.get("home_school_name")

        # away_team, away_rank = _split_rank(away_raw)
        # home_team, home_rank = _split_rank(home_raw)
        # Prefer explicit rank columns if available
        # away_rank = row_data.get("visitor_rank", "").strip()
        # home_rank = row_data.get("home_rank", "").strip()

        # # Fallback to parsing from name if needed
        # away_team, parsed_away_rank = _split_rank(away_raw)
        # home_team, parsed_home_rank = _split_rank(home_raw)

        # if not away_rank:
        #     away_rank = parsed_away_rank
        # if not home_rank:
        #     home_rank = parsed_home_rank

        # Scores
        away_score = row_data.get("away_score")
        home_score = row_data.get("home_score")
        # away_score = row_data.get("visitor_pts", "").strip()
        # home_score = row_data.get("home_pts", "").strip()

        # Neutral site flag: look for an 'N' in the game_location or notes column
        location = row_data.get("game_location") or row_data.get("location") or ""
        neutral = "Y" if location.strip().upper() == "N" else "N"

        rows.append({
            "season":      year,
            "conference":  conf_name,
            "date":        date_val,
            "home_team":   home_team,
            # "home_rank":   home_rank,
            "away_team":   away_team,
            # "away_rank":   away_rank,
            "home_score":  home_score,
            "away_score":  away_score,
            "neutral":     neutral,
        })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "season", "conference", "date",
    "home_team", "home_rank", "away_team", "away_rank",
    "home_score", "away_score", "neutral",
]

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        season_label = f"{year - 1}-{str(year)[-2:]}"
        log.info("━━━ Season %s (year=%d) ━━━", season_label, year)

        out_path = OUTPUT_DIR / f"{year}.csv"
        if out_path.exists():
            log.info("  %s already exists — skipping.", out_path)
            continue

        conferences = get_conferences_for_season(year)
        if not conferences:
            log.warning("  No conferences found for %d. Skipping season.", year)
            continue

        all_games: list[dict] = []

        for conf_name, sched_url in conferences:
            log.info("  Conference: %s", conf_name)
            games = parse_schedule_page(conf_name, year, sched_url)
            log.info("    → %d games parsed.", len(games))
            all_games.extend(games)

        if all_games:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(all_games)
            log.info("  ✅ Wrote %d games to %s", len(all_games), out_path)
        else:
            log.warning("  No games collected for season %d.", year)

    log.info("Done! CSVs are in the '%s' folder.", OUTPUT_DIR)


if __name__ == "__main__":
    main()