from typing import Any, Dict


ACCEPTED_BATTLE_TYPES = {"ranked", "soloRanked"}


def is_supported_ranked_battle(battle: Dict[str, Any]) -> bool:
    battle_data = battle.get("battle", {})
    teams = battle_data.get("teams", [])
    if battle_data.get("type") not in ACCEPTED_BATTLE_TYPES:
        return False
    if len(teams) != 2:
        return False
    return all(len(team) == 3 for team in teams)


def get_players_from_ranked_battle(battle: Dict[str, Any]) -> list[tuple[str, str]]:
    players = []
    for team in battle["battle"]["teams"]:
        for player in team:
            players.append((player["tag"][1:], player["name"]))
    return players


def normalize_brawler_name(name: str) -> str:
    return name.strip().lower()
