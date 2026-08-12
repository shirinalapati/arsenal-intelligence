"""Pitcher team lookup via MLB Stats API."""

from __future__ import annotations

import numpy as np
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

MLB_ABBRS = set(MLB_NAME_TO_ABBR.values())

# Statcast/Savant abbreviations → dashboard filter abbreviations
STATCAST_TO_ABBR = {
    "AZ": "ARI",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "KC": "KCR",
    "WSH": "WSN",
    "CWS": "CHW",
}


def _normalize_abbr(abbr: str | None) -> str | None:
    if not abbr:
        return None
    up = str(abbr).upper()
    return STATCAST_TO_ABBR.get(up, up)


def _team_from_split(stat: dict) -> str | None:
    team = stat.get("team") or {}
    abbr = team.get("abbreviation") or team.get("teamCode")
    if abbr:
        return _normalize_abbr(str(abbr))
    name = team.get("name")
    if name in MLB_NAME_TO_ABBR:
        return MLB_NAME_TO_ABBR[name]
    return None


def _team_from_person(person: dict) -> str | None:
    stats_list = person.get("stats") or []
    for stats in stats_list:
        for split in stats.get("splits") or []:
            abbr = _team_from_split(split)
            if abbr:
                return abbr
    current = person.get("currentTeam") or {}
    name = current.get("name")
    if name in MLB_NAME_TO_ABBR:
        return MLB_NAME_TO_ABBR[name]
    return _team_from_split({"team": current})


def _pitcher_team_from_statcast_rows(sub: pd.DataFrame) -> str:
    if sub.empty or "inning_topbot" not in sub.columns:
        return ""
    pitching = np.where(
        sub["inning_topbot"].eq("Top"),
        sub["home_team"],
        sub["away_team"],
    )
    teams = pd.Series(pitching, index=sub.index).map(lambda t: _normalize_abbr(str(t)) or "")
    teams = teams[teams != ""]
    if teams.empty:
        return ""
    grp = (
        pd.DataFrame({"team": teams, "game_date": sub.loc[teams.index, "game_date"]})
        .groupby("team")
        .agg(n=("team", "size"), last=("game_date", "max"))
        .sort_values(["last", "n"], ascending=[False, False])
    )
    ordered = [t for t in grp.index.tolist() if t in MLB_ABBRS]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    if grp.loc[ordered[1], "n"] >= max(50, grp.loc[ordered[0], "n"] * 0.05):
        return "/".join(ordered[:2])
    return ordered[0]


def teams_from_statcast(statcast_df: pd.DataFrame, season: int) -> dict[int, str]:
    """Derive MLB team(s) from pitch-level home/away + inning half."""
    need = {"pitcher", "home_team", "away_team", "inning_topbot", "game_date", "season"}
    if statcast_df.empty or not need.issubset(statcast_df.columns):
        return {}
    sub = statcast_df[statcast_df["season"] == season]
    out: dict[int, str] = {}
    for pid, grp in sub.groupby("pitcher"):
        team = _pitcher_team_from_statcast_rows(grp)
        if team:
            out[int(pid)] = team
    return out


def fetch_pitcher_teams(pitcher_ids: list[int], season: int) -> dict[int, str]:
    ids = sorted({int(x) for x in pitcher_ids if pd.notna(x) and int(x) > 0})
    out: dict[int, str] = {}
    hydrate = f"currentTeam,stats(group=pitching,type=season,season={season})"
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
            abbr = _team_from_person(person)
            if abbr:
                out[int(pid)] = abbr
    return out


def fill_missing_teams(
    df: pd.DataFrame,
    statcast_df: pd.DataFrame | None = None,
    pitcher_col: str = "pitcher",
    season_col: str = "season",
) -> pd.DataFrame:
    """Fill blank team values using MLB API, then Statcast pitch data."""
    out = df.copy()
    if "team" not in out.columns:
        out["team"] = ""
    out["team"] = out["team"].fillna("").astype(str)

    missing = out["team"].str.strip() == ""
    if missing.any():
        filled = attach_teams(out.loc[missing].drop(columns=["team"], errors="ignore"))
        out.loc[missing, "team"] = filled["team"].fillna("").astype(str).values

    missing = out["team"].str.strip() == ""
    if missing.any() and statcast_df is not None and not statcast_df.empty:
        for season in out.loc[missing, season_col].unique():
            sc_map = teams_from_statcast(statcast_df, int(season))
            if not sc_map:
                continue
            idx = out.index[missing & (out[season_col] == season)]
            for row_idx in idx:
                pid = int(out.at[row_idx, pitcher_col])
                if pid in sc_map:
                    out.at[row_idx, "team"] = sc_map[pid]

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
    team_frames = []
    for season, grp in df.groupby(season_col):
        mapping = fetch_pitcher_teams(grp[pitcher_col].astype(int).tolist(), int(season))
        part = grp.copy()
        part["team"] = part[pitcher_col].astype(int).map(lambda p: mapping.get(int(p), ""))
        team_frames.append(part)
    return pd.concat(team_frames, ignore_index=True)
