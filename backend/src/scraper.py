import json
import os
from pathlib import Path

import requests

from db import Database
from datasource import DevBrawlAPI


OUT_DIR = Path("./out")
BRAWLERS_DIR = OUT_DIR / "brawlers"
BRAWLIFY_BASE_URL = "https://api.brawlify.com/v1"
ALLOWED_GAME_MODES = {
    "Bounty",
    "Brawl Ball",
    "Brawl Hockey",
    "Cleaning Duty",
    "Gem Grab",
    "Heist",
    "Hot Zone",
    "Knockout",
}


def _ensure_out_dir() -> None:
    BRAWLERS_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_json(url: str, *, headers: dict | None = None) -> dict:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_supercell_brawlers() -> list[dict]:
    api = DevBrawlAPI()
    data = api.get_brawler_information()
    if not data or "items" not in data:
        raise RuntimeError("Failed to fetch brawler information from Supercell")
    return sorted(data["items"], key=lambda item: int(item["id"]))


def fetch_brawlify_brawlers() -> dict[str, dict]:
    data = _fetch_json(f"{BRAWLIFY_BASE_URL}/brawlers")
    return {
        item["name"].lower(): item
        for item in data.get("list", [])
        if item.get("name")
    }


def fetch_brawlify_maps() -> list[dict]:
    data = _fetch_json(f"{BRAWLIFY_BASE_URL}/maps")
    return data.get("list", [])


def build_brawler_catalog() -> dict[str, dict]:
    supercell_brawlers = fetch_supercell_brawlers()
    brawlify_brawlers = fetch_brawlify_brawlers()

    catalog = {}
    for index, brawler in enumerate(supercell_brawlers):
        name = brawler["name"].lower()
        enrichment = brawlify_brawlers.get(name, {})
        catalog[name] = {
            "index": index,
            "name": name,
            "supercell_id": int(brawler["id"]),
            "class": enrichment.get("class", {}).get("name", "Unknown"),
            "rarity": enrichment.get("rarity", {}).get("name", "Unknown"),
            "image_url": enrichment.get("imageUrl"),
        }

    return catalog


def save_brawler_data(brawler_data: dict[str, dict]) -> None:
    _ensure_out_dir()
    with open(BRAWLERS_DIR / "brawlers.json", "w") as f:
        json.dump(brawler_data, f, indent=2)


def cache_stripped_brawler_data() -> None:
    with open(BRAWLERS_DIR / "brawlers.json", "r") as f:
        brawler_data = json.load(f)

    classes = sorted({stats["class"] for stats in brawler_data.values()})
    class_to_idx = {brawler_class: idx for idx, brawler_class in enumerate(classes)}

    stripped_brawler_data = {}
    for brawler_name, brawler_info in brawler_data.items():
        stripped_brawler_data[brawler_name] = {
            "class_idx": class_to_idx[brawler_info["class"]],
            "index": brawler_info["index"],
        }

    with open(BRAWLERS_DIR / "stripped_brawlers.json", "w") as f:
        json.dump(stripped_brawler_data, f, indent=2)


def brawler_to_supercell_id_mapping() -> None:
    with open(BRAWLERS_DIR / "brawlers.json", "r") as f:
        brawler_data = json.load(f)

    brawler_id_dict = {
        brawler_name: int(brawler_info["supercell_id"])
        for brawler_name, brawler_info in brawler_data.items()
    }

    with open(BRAWLERS_DIR / "brawler_supercell_id_mapping.json", "w") as f:
        json.dump(brawler_id_dict, f, indent=2)


def _get_canonical_brawler_names() -> list[str]:
    with open(BRAWLERS_DIR / "stripped_brawlers.json", "r") as f:
        return list(json.load(f).keys())


def cache_brawler_winrates() -> None:
    print("Retrieving winrates for all brawlers for each map.")
    db = Database()
    try:
        brawler_names = _get_canonical_brawler_names()
        map_winrates = {}
        for map_row in db.get_all_maps():
            map_name = map_row[0]
            map_winrates[map_name] = {
                brawler_name: db.check_brawler_winrate_for_map(
                    brawler=brawler_name,
                    map_name=map_name,
                )
                for brawler_name in brawler_names
            }

        with open(BRAWLERS_DIR / "brawler_winrates.json", "w") as f:
            json.dump(map_winrates, f, indent=2)
    finally:
        db.close()


def cache_brawler_pickrates() -> None:
    print("Retrieving pickrates for all brawlers for each map.")
    db = Database()
    try:
        brawler_names = _get_canonical_brawler_names()
        map_pickrates = {}
        for map_row in db.get_all_maps():
            map_name = map_row[0]
            map_pickrates[map_name] = {
                brawler_name: db.check_brawler_significance_for_map(
                    brawler_name,
                    map_name,
                )
                for brawler_name in brawler_names
            }

        with open(BRAWLERS_DIR / "brawler_pickrates.json", "w") as f:
            json.dump(map_pickrates, f, indent=2)
    finally:
        db.close()


def save_map_data() -> None:
    _ensure_out_dir()
    map_data = {}
    for brawl_map in fetch_brawlify_maps():
        game_mode = brawl_map.get("gameMode", {}).get("name")
        if game_mode not in ALLOWED_GAME_MODES:
            continue
        map_name = brawl_map.get("name")
        if not map_name:
            continue
        map_data[map_name] = {
            "img_url": brawl_map.get("imageUrl"),
            "game_mode": game_mode,
        }

    with open(BRAWLERS_DIR / "map_data.json", "w") as f:
        json.dump(map_data, f, indent=2)


def sync_metadata() -> None:
    _ensure_out_dir()
    catalog = build_brawler_catalog()
    save_brawler_data(catalog)
    cache_stripped_brawler_data()
    brawler_to_supercell_id_mapping()
    save_map_data()


def main() -> None:
    sync_metadata()
    cache_brawler_winrates()
    cache_brawler_pickrates()


if __name__ == "__main__":
    main()
