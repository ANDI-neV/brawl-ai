import requests
from bs4 import BeautifulSoup
import pandas as pd
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
    label_dict = {"normal": float(values[0])}

    for value, label in zip(values[1:], labels[1:]):
        condition = label.replace("with", "").strip()
        label_dict[condition] = float(value)
    return label_dict

def get_brawler_data(brawler_name, brawler_url):
    response = requests.get(brawler_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    infobox = soup.select_one('aside.portable-infobox')
    brawler_info = {"name": brawler_name, "url": brawler_url}

    if infobox:
        for row in infobox.select('div.pi-data'):
            label = str.lower(row.select_one('h3').text.strip())
            value = row.select_one('div.pi-data-value').text.strip()

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
                brawler_info["hypercharge"][label] = value
            else:
                brawler_info[label] = parse_label_info(value)


    return brawler_info

def save_brawler_data(brawler_data):
    with open(os.path.join(BRAWLERS_DIR, 'brawlers.json'), 'w') as f:
        json.dump(brawler_data, f, indent=2)

def main():
    brawler_list = get_brawler_list()
    brawler_data = []

    for brawler_name, brawler_url in brawler_list:
        print(f"Fetching data for {brawler_name}")
        brawler_details = get_brawler_data(brawler_name, brawler_url)
        brawler_data.append(brawler_details)

    save_brawler_data(brawler_data)
    print("Brawler data successfully fetched and saved.")

if __name__ == "__main__":
    main()