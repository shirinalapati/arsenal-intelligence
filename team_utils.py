"""Pitcher team lookup via MLB Stats API."""

from __future__ import annotations

import pandas as pd
import requests

MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
BATCH = 50

MLB_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYY",
    "New York Mets": "NYM",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def _team_from_split(stat: dict) -> str | None:
    team = stat.get("team") or {}
    abbr = team.get("abbreviation") or team.get("teamCode")
    if abbr:
        return str(abbr).upper()
    name = team.get("name")
    if name in MLB_NAME_TO_ABBR:
        return MLB_NAME_TO_ABBR[name]
    return None


def fetch_pitcher_teams(pitcher_ids: list[int], season: int) -> dict[int, str]:
    ids = sorted({int(x) for x in pitcher_ids if pd.notna(x) and int(x) > 0})
    out: dict[int, str] = {}
    hydrate = f"stats(group=pitching,type=season,season={season})"
    for i in range(0, len(ids), BATCH):
        batch = ids[i : i + BATCH]
        resp = requests.get(
            MLB_PEOPLE_URL,
            params={"personIds": ",".join(str(x) for x in batch), "hydrate": hydrate},
            timeout=90,
        )
        resp.raise_for_status()
        for person in resp.json().get("people") or []:
            pid = person.get("id")
            if pid is None:
                continue
            stats_list = person.get("stats") or []
            if not stats_list:
                continue
            splits = stats_list[0].get("splits") or []
            if not splits:
                continue
            abbr = _team_from_split(splits[0])
            if abbr:
                out[int(pid)] = abbr
    return out


def fetch_pitcher_roles_2026(pitcher_ids: list[int], season: int = 2026) -> dict[int, str]:
    """Classify SP vs RP from MLB games started / games pitched."""
    ids = sorted({int(x) for x in pitcher_ids if pd.notna(x) and int(x) > 0})
    out: dict[int, str] = {}
    hydrate = f"stats(group=pitching,type=season,season={season})"
    for i in range(0, len(ids), BATCH):
        batch = ids[i : i + BATCH]
        resp = requests.get(
            MLB_PEOPLE_URL,
            params={"personIds": ",".join(str(x) for x in batch), "hydrate": hydrate},
            timeout=90,
        )
        resp.raise_for_status()
        for person in resp.json().get("people") or []:
            pid = person.get("id")
            if pid is None:
                continue
            stats_list = person.get("stats") or []
            if not stats_list:
                continue
            stat = (stats_list[0].get("splits") or [{}])[0].get("stat") or {}
            gs = float(stat.get("gamesStarted") or 0)
            g = float(stat.get("gamesPlayed") or stat.get("gamesPitched") or 0)
            if gs >= 5 or (g > 0 and gs / g >= 0.5):
                out[int(pid)] = "SP"
            elif g > 0:
                out[int(pid)] = "RP"
    return out


def attach_teams(df: pd.DataFrame, pitcher_col: str = "pitcher", season_col: str = "season") -> pd.DataFrame:
    """Add `team` column to a pitcher-season dataframe."""
    if df.empty:
        df = df.copy()
        df["team"] = ""
        return df
    out = df.copy()
    teams: list[str] = []
    for season, grp in out.groupby(season_col):
        mapping = fetch_pitcher_teams(grp[pitcher_col].astype(int).tolist(), int(season))
        teams.extend(grp[pitcher_col].astype(int).map(lambda p: mapping.get(int(p), "")).tolist())
    # groupby order fix — merge properly
    team_frames = []
    for season, grp in out.groupby(season_col):
        mapping = fetch_pitcher_teams(grp[pitcher_col].astype(int).tolist(), int(season))
        part = grp.copy()
        part["team"] = part[pitcher_col].astype(int).map(lambda p: mapping.get(int(p), ""))
        team_frames.append(part)
    return pd.concat(team_frames, ignore_index=True)
