import json
import os
import string
from typing import Dict, List

import requests
from requests.exceptions import HTTPError

from settings import get_api_token


BRAWLERS_JSON_PATH = "out/brawlers/stripped_brawlers.json"
BRAWLER_WINRATES_JSON_PATH = "out/brawlers/brawler_winrates.json"
BRAWLER_PICKRATES_JSON_PATH = "out/brawlers/brawler_pickrates.json"

picking_combinations1 = [["a1", "b1", "b2", "a2"],
                         ["a1", "b1", "b2", "a2", "a3"]]
picking_combinations2 = [["b1", "a1"],
                         ["b1", "a1", "a2"],
                         ["b1", "a1", "a2", "b2", "b3", "a3"]]

first_pick_sequence = ["a1", "b1", "b2", "a2", "a3", "b3"]
second_pick_sequence = ["b1", "a1", "a2", "b2", "b3", "a3"]


class PlayerNotFoundError(Exception):
    pass


def _load_json(path: str):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path), "r") as json_file:
        return json.load(json_file)


def prepare_brawler_data() -> Dict:
    return _load_json(BRAWLERS_JSON_PATH)


def initialize_brawler_data():
    brawler_data = prepare_brawler_data()
    n_brawlers = len(brawler_data)
    n_classes = max(
        [brawler["class_idx"] for brawler in brawler_data.values()],
        default=0,
    )
    constants = {
        "n_brawlers": n_brawlers,
        "n_classes": n_classes,
        "CLASS_CLS_TOKEN_INDEX": n_classes + 1,
        "CLASS_PAD_TOKEN_INDEX": n_classes + 2,
        "CLS_TOKEN_INDEX": n_brawlers,
        "PAD_TOKEN_INDEX": n_brawlers + 1,
        "n_brawlers_with_special_tokens": n_brawlers + 2,
        "index_to_brawler_name": {
            data["index"]: name for name, data in brawler_data.items()
        },
    }
    return brawler_data, constants


def get_all_brawlers():
    brawler_data = prepare_brawler_data()
    return list(brawler_data.keys())


def load_map_id_mapping():
    with open("out/models/map_id_mapping.json", "r") as mapping_file:
        map_id_mapping = json.load(mapping_file)
    return {str(key): value for key, value in map_id_mapping.items()}


def get_map_winrate(map_name):
    return _load_json(BRAWLER_WINRATES_JSON_PATH)[map_name]


def get_map_pickrate(map_name):
    return _load_json(BRAWLER_PICKRATES_JSON_PATH)[map_name]


def get_map_score(map_name):
    pick_rate = dict(get_map_pickrate(map_name))
    winrate = dict(get_map_winrate(map_name))
    return {
        brawler: float(pick_rate[brawler]) * float(winrate[brawler])
        for brawler in pick_rate
    }


def get_filtered_brawlers(player_tag, min_level):
    if player_tag.startswith("#"):
        player_tag = player_tag[1:]

    response = requests.get(
        f"https://api.brawlstars.com/v1/players/%23{player_tag}",
        headers={"Authorization": f"Bearer {get_api_token()}"},
        timeout=20,
    )

    try:
        response.raise_for_status()
    except HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise PlayerNotFoundError("Player tag not found") from exc
        raise

    data = response.json()
    return [
        str.lower(brawler["name"])
        for brawler in data.get("brawlers", [])
        if brawler.get("power", 0) >= min_level
    ]


def get_brawler_index(brawler_name):
    brawler_data, _ = initialize_brawler_data()
    return brawler_data.get(str.lower(brawler_name), {}).get("index")


def get_brawler_class(brawler_name):
    brawler_data, _ = initialize_brawler_data()
    return brawler_data.get(str.lower(brawler_name), {}).get("class_idx")


def acquire_combination(brawler_dict, first_pick):
    possible_combinations = (
        picking_combinations1 if first_pick else picking_combinations2
    )
    for combination in possible_combinations:
        if (
            all(pos in brawler_dict for pos in combination[:-1])
            and len(combination[:-1]) == len(brawler_dict)
        ):
            return combination
    return None


def get_brawler_dict(picks: List[str], first_pick: bool) -> Dict[str, str]:
    sequence = first_pick_sequence if first_pick else second_pick_sequence
    return {sequence[index]: brawler for index, brawler in enumerate(picks)}


def get_all_maps():
    with open("out/brawlers/map_data.json", "r") as all_maps_file:
        all_maps = json.load(all_maps_file)

    current_maps = load_map_id_mapping()
    filtered_maps = {}
    for active_map in current_maps.keys():
        if active_map in all_maps:
            filtered_maps[active_map] = all_maps[active_map]
            continue

        normalized_name = string.capwords(active_map).replace("'", "")
        if normalized_name in all_maps:
            filtered_maps[active_map] = all_maps[normalized_name]

    return filtered_maps
