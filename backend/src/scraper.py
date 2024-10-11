import requests
from bs4 import BeautifulSoup
import json
import os
import re
from typing import Any
from db import Database
from datasource import DevBrawlAPI

BASE_URL = "https://brawlstars.fandom.com"
BRAWLERS_CATEGORY_URL = f"{BASE_URL}/wiki/Category:Brawlers"
OUT_DIR = "./out"
BRAWLERS_DIR = os.path.join(OUT_DIR, "brawlers")


def get_brawler_list() -> list[Any]:
    response = requests.get(BRAWLERS_CATEGORY_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    brawler_links = soup.select("div.category-page__members "
                                "a.category-page__member-link")
    brawler_links = list(
        filter(lambda link: 'Category' not in link['href'],
               brawler_links))

    brawler_names = list(
        filter(lambda k: 'Category' not in k,
               [str.lower(link.text.strip())
                for link in brawler_links]))

    brawler_urls = [BASE_URL + link['href']
                    for link in brawler_links]
    return list(zip(brawler_names, brawler_urls))


def parse_label_info(value):
    values = re.findall(r'\d+\.?\d*', value)
    labels = [label.lower().strip("()") for label
              in re.findall(r'\((.*?)\)', value)]
    label_dict = {}

    if len(values) > len(labels):
        labels = ["normal"] + labels
    else:
        labels[0] = "normal"

    for value, label in zip(values, labels):
        condition = label.replace("with", "").strip()
        label_dict[condition] = float(value)
    return label_dict


def convert_percent_to_float(value):
    return float(value.strip('%')) / 100


def extract_text_with_br(element):
    return ' BREAK '.join(element.stripped_strings)


def add_supercell_brawler_indices(brawler_dict):
    api = DevBrawlAPI()
    data = api.get_brawler_information()
    for brawler in data:
        brawler_dict[brawler["name"]]["supercell_id"] = brawler["id"]


def brawler_to_supercell_id_mapping():
    api = DevBrawlAPI()
    data = api.get_brawler_information()
    print(data)
    brawler_id_dict = {}
    for brawler in data['items']:
        brawler_id_dict[str.lower(brawler['name'])] = int(brawler['id'])

    with open(os.path.join(BRAWLERS_DIR, 'brawler_supercell_id_mapping.json'), 'w') as f:
        json.dump(brawler_id_dict, f, indent=2)


def get_brawler_data(brawler_name, brawler_url, index):
    response = requests.get(brawler_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    infobox = soup.select_one('aside.portable-infobox')
    brawler_info = {"name": brawler_name, "url": brawler_url, "index": index}

    if infobox:
        for row in infobox.select('div.pi-data'):
            label = str.lower(row.select_one('h3').text.strip())
            value_element = row.select_one('div.pi-data-value')
            value = extract_text_with_br(value_element)
            excluded_labels = ["release date", "voice actor"]
            string_labels = ["rarity", "class"]
            hyper_labels = ["hypercharge multiplier", "speed",
                            "damage", "shield"]

            if label in excluded_labels:
                continue

            if label in string_labels:
                brawler_info[label] = value
            elif label in hyper_labels:
                if "hypercharge" not in brawler_info:
                    brawler_info["hypercharge"] = {}
                brawler_info["hypercharge"][label] = (
                    convert_percent_to_float(value.replace("+", "").strip()))
            elif label not in brawler_info:
                brawler_info[label] = parse_label_info(value)

    level_tables = soup.find_all('table', class_='pi-horizontal-group')
    brawler_info["level stats"] = {}
    for table in level_tables:
        rows = table.find_all('tr')
        for row in rows:
            columns = row.find_all('td')

            for i in range(len(columns[1:])):
                data_source = str.lower(columns[i + 1].get('data-source'))
                if (data_source not in brawler_info["level stats"]
                        and columns[i + 1].text.strip().isdigit()):
                    brawler_info["level stats"][data_source] = {}
                try:
                    level = int(columns[0].text.strip())
                    value = int(columns[i + 1].text.strip())

                    brawler_info["level stats"][data_source][level] = value
                except ValueError:
                    # Skip rows that don't contain valid integers
                    continue

    return brawler_info


def save_brawler_data(brawler_data):
    with open(os.path.join(BRAWLERS_DIR, 'brawlers.json'), 'w') as f:
        json.dump(brawler_data, f, indent=2)


def get_highest_brawler_index(brawler_data):
    return max([brawler.get('index', 0)
                for brawler in brawler_data.values()], default=0)


def main():
    brawler_list = get_brawler_list()
    brawler_data = {}

    if not os.path.exists(BRAWLERS_DIR):
        os.makedirs(BRAWLERS_DIR)

    if os.path.exists(os.path.join(BRAWLERS_DIR, 'brawlers.json')):
        with (open(os.path.join(BRAWLERS_DIR, 'brawlers.json'), 'r')
              as json_file):
            loaded_data = json.load(json_file)
            if isinstance(loaded_data, dict):
                brawler_data = loaded_data
            else:
                print("Warning: Loaded data is not a dictionary. "
                      "Resetting to an empty dictionary.")
                brawler_data = {}

    for brawler_name, brawler_url in brawler_list:
        print(f"Fetching data for {brawler_name}")

        if brawler_name in brawler_data:
            brawler_details = (
                get_brawler_data(brawler_name,
                                 brawler_url,
                                 brawler_data[brawler_name].get('index')))
        else:
            brawler_details = (
                get_brawler_data(brawler_name,
                                 brawler_url,
                                 get_highest_brawler_index(brawler_data) + 1))

        brawler_data[str.lower(brawler_name)] = brawler_details

    cache_brawler_winrates(brawler_list)
    cache_brawler_pickrates(brawler_list)

    save_brawler_data(brawler_data)
    print("Brawler data successfully fetched and saved.")

    scrape_brawler_images(brawler_data)


def scrape_brawler_images(brawler_data):
    images_dir = os.path.join(OUT_DIR, "brawler_images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for brawler_name, brawler_info in brawler_data.items():
        scrape_brawler_image(brawler_name, brawler_info['url'],
                             images_dir)


def scrape_brawler_image(brawler_name, url, images_dir):
    print(f"Scraping image for {brawler_name}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    image = soup.select_one('img.lazyload[alt*="Portrait"]')

    if image and 'data-src' in image.attrs:
        image_url = image['data-src']
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            image_path = os.path.join(images_dir, f"{brawler_name}.png")
            with open(image_path, 'wb') as img_file:
                img_file.write(img_response.content)
            print(f"Image saved for {brawler_name}")
        else:
            print(f"Failed to download image for {brawler_name}")
    else:
        print(f"No image found for {brawler_name}")


def cache_brawler_winrates(brawler_list: list[Any]):
    print("Retrieving winrates for all brawlers for each map.")
    db = Database()

    map_winrates = {}
    maps = db.get_all_maps()
    for map in maps:
        brawler_winrates = {}
        map = map[0]
        for brawler_name, brawler_url in brawler_list:
            winrate = (
                db.check_brawler_winrate_for_map(brawler=brawler_name.upper(),
                                                 map_name=map))
            print(f"Winrate for {brawler_name} on {map}: {winrate}")
            brawler_winrates[brawler_name] = winrate
        map_winrates[map] = brawler_winrates

    with open(os.path.join(BRAWLERS_DIR, 'brawler_winrates.json'), 'w') as f:
        json.dump(map_winrates, f, indent=2)


def cache_brawler_pickrates(brawler_list: list[Any]):
    print("Retrieving pickrates for all brawlers for each map.")
    db = Database()

    map_pickrates = {}
    maps = db.get_all_maps()
    for map in maps:
        brawler_significance = {}
        map = map[0]
        for brawler_name, brawler_url in brawler_list:
            pickrate = (
                db.check_brawler_significance_for_map(brawler_name.upper(),
                                                      map))
            print(f"Pickrate for {brawler_name} on {map}: {pickrate}")
            brawler_significance[brawler_name] = pickrate
        map_pickrates[map] = brawler_significance

    with open(os.path.join(BRAWLERS_DIR, 'brawler_pickrates.json'), 'w') as f:
        json.dump(map_pickrates, f, indent=2)

#/html/body/div[2]/div[2]/div[2]/div/div[2]/div[2]/a


def scrape_map_data():
    response = requests.get("https://brawlify.com/maps/")
    soup = BeautifulSoup(response.text, 'html.parser')

    map_data = []

    # Find all game mode sections
    game_mode_sections = soup.find_all('div', class_='row mb-4 align-items-center justify-content-center')

    for section in game_mode_sections:
        # Extract game mode
        game_mode_element = section.find('h2', class_='h3 recomm-mode-text pt-2')
        if game_mode_element:
            game_mode = game_mode_element.text.split('(')[0].strip()

            # Find all maps for this game mode
            map_elements = section.find_all('div', class_='map-def')

            for map_element in map_elements:
                map_name = map_element.find('span', class_='badge map-name').text
                map_image = map_element.find('img')['src']

                map_data.append({
                    'game_mode': game_mode,
                    'map_name': map_name,
                    'map_image': map_image
                })

    return map_data


def save_map_data():
    scraped_data = scrape_map_data()
    map_data = {}
    for item in scraped_data:
        map_data[item['map_name']] = {}
        map_data[item['map_name']]['img_url'] = item['map_image']
        map_data[item['map_name']]['game_mode'] = item['game_mode']

    with open(os.path.join(BRAWLERS_DIR, 'map_data.json'), 'w') as f:
        json.dump(map_data, f, indent=2)


def test_map_data():
    scraped_data = scrape_map_data()
    for item in scraped_data:
        print(f"Game Mode: {item['game_mode']}")
        print(f"Map Name: {item['map_name']}")
        print(f"Map Image URL: {item['map_image']}")
        print("---")


if __name__ == "__main__":
    brawler_list = get_brawler_list()
    cache_brawler_pickrates(brawler_list)
    cache_brawler_winrates(brawler_list)
