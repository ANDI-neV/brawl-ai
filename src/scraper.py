import requests
from bs4 import BeautifulSoup
import json
import os
import re

BASE_URL = "https://brawlstars.fandom.com"
BRAWLERS_CATEGORY_URL = f"{BASE_URL}/wiki/Category:Brawlers"
OUT_DIR = "./out"
BRAWLERS_DIR = os.path.join(OUT_DIR, "brawlers")

def get_brawler_list():
    repsonse = requests.get(BRAWLERS_CATEGORY_URL)
    soup = BeautifulSoup(repsonse.text, 'html.parser')
    brawler_links = soup.select("div.category-page__members a.category-page__member-link")
    brawler_links = list(filter(lambda link: 'Category' not in link['href'], brawler_links))
    brawler_names = list(filter(lambda k: 'Category' not in k, [link.text.strip() for link in brawler_links]))
    brawler_urls = [BASE_URL + link['href'] for link in brawler_links]
    return list(zip(brawler_names, brawler_urls))

def parse_label_info(value):
    values = re.findall(r'\d+\.?\d*', value)
    labels = [label.lower().strip("()") for label in re.findall(r'\((.*?)\)', value)]
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
            hyper_labels = ["hypercharge multiplier", "speed", "damage", "shield"]

            if label in excluded_labels:
                continue

            if label in string_labels:
                brawler_info[label] = value
            elif label in hyper_labels:
                if "hypercharge" not in brawler_info:
                    brawler_info["hypercharge"] = {}
                brawler_info["hypercharge"][label] = convert_percent_to_float(value.replace("+","").strip())
            else:
                brawler_info[label] = parse_label_info(value)

    level_tables = soup.find_all('table', class_='pi-horizontal-group')
    brawler_info["level stats"] = {}
    for table in level_tables:
        rows = table.find_all('tr')
        for row in rows:
            columns = row.find_all('td')

            for i in range(len(columns[1:])):
                data_source = str.lower(columns[i+1].get('data-source'))
                if data_source not in brawler_info["level stats"] and columns[i+1].text.strip().isdigit():
                    brawler_info["level stats"][data_source] = {}
                try:
                    level = int(columns[0].text.strip())
                    value = int(columns[i+1].text.strip())

                    brawler_info["level stats"][data_source][level] = value
                except ValueError:
                    # Skip rows that don't contain valid integers
                    continue

    return brawler_info

def save_brawler_data(brawler_data):
    with open(os.path.join(BRAWLERS_DIR, 'brawlers.json'), 'w') as f:
        json.dump(brawler_data, f, indent=2)

def get_highest_brawler_index():
    return int(lambda brawler_data: max([brawler.get('index') for brawler in brawler_data]))

def main():
    brawler_list = get_brawler_list()
    brawler_data = []

    index = 0
    for brawler_name, brawler_url in brawler_list:
        print(f"Fetching data for {brawler_name}")
        with open('./out/brawlers/brawlers.json', 'r') as json_file:
            data = json.load(json_file)

        brawler_found = False
        for entry in data:
            if entry.get('name') == brawler_name:
                brawler_found = True
                index = entry.get('index')
                break

        if brawler_found:
            brawler_details = get_brawler_data(brawler_name, brawler_url, index)
        else:
            brawler_details = get_brawler_data(brawler_name, brawler_url, get_highest_brawler_index() + 1)

        brawler_data.append(brawler_details)
        index += 1

    save_brawler_data(brawler_data)
    print("Brawler data successfully fetched and saved.")

if __name__ == "__main__":
    main()