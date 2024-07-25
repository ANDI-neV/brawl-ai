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
    engine = create_engine('sqlite:///../games.db')

    match_data = pd.read_sql_query("SELECT * FROM battles WHERE mode='gemGrab'", con=engine)

    match_data = match_data.drop(["id", "battleTime", "mode", "map"], axis=1)

    with open('./out/brawlers/brawlers.json', 'r') as json_file:
        brawler_data = json.load(json_file)

    return match_data, brawler_data

def get_brawler_features(brawler_name, brawler_data):
    print("brawler name:" + brawler_name)
    print(brawler_data[brawler_name])
    brawler = brawler_data[brawler_name]
    features = {
        'movement_speed_normal': brawler['movement speed']['normal'],
        'range_normal': brawler['range']['normal'],
        'reload_normal': brawler['reload']['normal'],
        'projectile_speed_normal': brawler['projectile speed']['normal'],
    }
    return features

def create_brawler_matrix(match_data, brawler_data):
    matrix = np.zeros((len(match_data), 24))
    print(matrix)
    for i in range(1, 4):
        for team in ['a', 'b']:
            brawler_col = f'{team}{i}'
            brawler_name = str.lower(match_data[brawler_col])
            print(brawler_name)
            features = get_brawler_features(brawler_name, brawler_data)
            matrix = np.append(matrix, list(features.values()))
            print(matrix)

    return matrix, match_data['result']


match_data, brawler_data = prepare_training_data()
print(match_data.iloc[0])
print(brawler_data['shelly'])
create_brawler_matrix(match_data.iloc[0], brawler_data)