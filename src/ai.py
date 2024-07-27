# we do stuff here
import pandas as pd
import os
import json
import numpy as np
from sqlalchemy import create_engine
from db import Database

def prepare_training_data():
    engine = create_engine('sqlite:///../games.db')

    match_data = pd.read_sql_query("SELECT * FROM battles WHERE mode='knockout'", con=engine)

    match_data = match_data.drop(["id", "battleTime", "mode", "map"], axis=1)

    with open('./out/brawlers/brawlers.json', 'r') as json_file:
        brawler_data = json.load(json_file)

    return match_data, brawler_data

def get_brawler_features(brawler_name, brawler_data):
    brawler = brawler_data[brawler_name]
    features = {
        'index': brawler['index'],
        'movement_speed_normal': brawler['movement speed']['normal'],
        'range_normal': brawler['range']['normal'],
        #'reload_normal': brawler['reload']['normal'], reload feature is only available for 80/82 brawlers
        #'projectile_speed_normal': brawler['projectile speed']['normal'], projectile speed feature is only available for 81/82 brawlers
        f'health': brawler['level stats']['health']['11'],
    }
    return features


def create_brawler_matrix(match_data, brawler_data, limit=None):
    vectors = []
    results = []

    if limit is not None:
        match_data = match_data.head(limit)

    for index, row in match_data.iterrows():
        match_vector = []
        for i in range(1, 4):
            for team in ['a', 'b']:
                brawler_col = f'{team}{i}'
                brawler_name = str.lower(row[brawler_col])
                features = get_brawler_features(brawler_name, brawler_data)
                vector = list(features.values())
                match_vector.extend(vector)

        print(match_vector)
        vectors.append(match_vector)
        results.append(row['result'])

    return np.array(vectors), np.array(results)

match_data, brawler_data = prepare_training_data()
print(match_data.head())
print("amount of battles: ", len(match_data))

X, y = create_brawler_matrix(match_data, brawler_data)

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
