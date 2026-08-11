"""Team filter helpers for the Streamlit dashboard."""

MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

ABBR_ALIASES: dict[str, list[str]] = {
    "ARI": ["ARI", "Arizona"],
    "ATH": ["ATH", "OAK", "Athletics"],
    "ATL": ["ATL", "Atlanta"],
    "BAL": ["BAL", "Baltimore"],
    "BOS": ["BOS", "Boston"],
    "CHC": ["CHC", "Chicago Cubs"],
    "CHW": ["CHW", "Chicago White Sox"],
    "CIN": ["CIN", "Cincinnati"],
    "CLE": ["CLE", "Cleveland"],
    "COL": ["COL", "Colorado"],
    "DET": ["DET", "Detroit"],
    "HOU": ["HOU", "Houston"],
    "KCR": ["KCR", "Kansas City"],
    "LAA": ["LAA", "Los Angeles Angels"],
    "LAD": ["LAD", "Los Angeles Dodgers"],
    "MIA": ["MIA", "Miami"],
    "MIL": ["MIL", "Milwaukee"],
    "MIN": ["MIN", "Minnesota"],
    "NYM": ["NYM", "New York Mets"],
    "NYY": ["NYY", "New York Yankees"],
    "PHI": ["PHI", "Philadelphia"],
    "PIT": ["PIT", "Pittsburgh"],
    "SDP": ["SDP", "San Diego"],
    "SEA": ["SEA", "Seattle"],
    "SFG": ["SFG", "San Francisco"],
    "STL": ["STL", "St. Louis"],
    "TBR": ["TBR", "Tampa Bay"],
    "TEX": ["TEX", "Texas"],
    "TOR": ["TOR", "Toronto"],
    "WSN": ["WSN", "Washington"],
}


def pitcher_matches_team(team_val: str, filter_abbr: str) -> bool:
    if not team_val or not filter_abbr or filter_abbr == "All":
        return True
    aliases = ABBR_ALIASES.get(filter_abbr.upper(), [filter_abbr.upper()])
    alias_set = {a.lower() for a in aliases}
    for part in str(team_val).split("/"):
        if part.strip().lower() in alias_set:
            return True
    return False
