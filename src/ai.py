# we do stuff here
import torch
import pandas as pd
import os
import json
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sqlalchemy import create_engine
from db import Database


def prepare_training_data():
    # Funktioniert nicht :(
    engine = create_engine('sqlite:///../games.db')

    match_data = pd.read_sql_query("SELECT * FROM battles WHERE mode='gemGrab'", con=engine)

    match_data = match_data.drop(["id", "battleTime", "mode", "map"], axis=1)

    with open('./out/brawlers/brawlers.json', 'r') as json_file:
        brawler_data = json.load(json_file)

    return match_data, brawler_data

def get_brawler_features(brawler_name, brawler_data):
    if brawler_name in brawler_data:
        brawler = brawler_data[brawler_name]
        features = {
            'movement_speed_normal': brawler['movement speed']['normal'],
            'range_normal': brawler['range']['normal'],
            'reload_normal': brawler['reload']['normal'],
            'projectiles_per_attack_normal': brawler['projectiles per attack']['normal'],
            'super_charge_per_hit_normal': brawler['super charge per hit']['normal'],
            'attack_spread_normal': brawler['attack spread']['normal'],
            'projectile_speed_normal': brawler['projectile speed']['normal'],
            'attack_width_normal': brawler['attack width']['normal'],
        }
        return features
    else:
        return None

def add_brawler_features_to_match_data(match_data, brawler_data):
    for i in range(1, 4):
        for team in ['a', 'b']:
            brawler_col = f'{team}{i}'
            features = get_brawler_features(match_data[brawler_col], brawler_data)
            if features:
                for feature, value in features.items():
                    match_data[f'{brawler_col}_{feature}'] = value

match_data, brawler_data = prepare_training_data()
add_brawler_features_to_match_data(match_data, brawler_data)