import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sqlalchemy import create_engine, URL
from torch.utils.data import TensorDataset, DataLoader
from db import Database
from sklearn.utils import class_weight


# Constants
BRAWLERS_JSON_PATH = 'out/brawlers/brawlers.json'
MODEL_PATH = 'out/models/out_in_the_open.pth'
ENCODER_PATH = 'out/models/encoder.joblib'
SCALER_PATH = 'out/models/scaler.joblib'
SHAPE_PATH = 'out/models/shape.joblib'

picking_combinations1 = [['a1'], ['a1', 'b1'], ['a1', 'b1', 'b2'], ['a1', 'b1', 'b2', 'a2'],
                         ['a1', 'b1', 'b2', 'a2', 'a3'], ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']]
picking_combinations1_new = [['a1', 'b1', 'b2', 'a2'], ['a1', 'b1', 'b2', 'a2', 'a3'],
                             ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']]
picking_combinations2 = [['b1'], ['b1', 'a1'], ['b1', 'a1', 'a2'], ['b1', 'a1', 'a2', 'b2'],
                         ['b1', 'a1', 'a2', 'b2', 'b3'], ['b1', 'a1', 'a2', 'b2', 'b3', 'a3']]
picking_combinations2_new = [['b1', 'a1'], ['b1', 'a1', 'a2'], ['b1', 'a1', 'a2', 'b2', 'b3', 'a3']]
all_players = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']


class BrawlStarsNN(nn.Module):
    def __init__(self, input_size: int):
        super(BrawlStarsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def prepare_training_data(map: str = None) -> pd.DataFrame:
    url_object = URL.create(
        "postgresql+psycopg2",
        username="REDACTED",
        password="REDACTED",
        host="REDACTED",
        database="REDACTED",
    )
    engine = create_engine(url_object)

    if map is None:
        query = "SELECT * FROM battles"
        match_data = pd.read_sql_query(query, con=engine)
    else:
        query = "SELECT * FROM battles WHERE map = %s"
        match_data = pd.read_sql_query(query, con=engine, params=(map,))

    return match_data


def get_dummy_vector(brawler_data, encoder, scaler, include_continuous_features):
    index_encoded = np.zeros(len(brawler_data))
    if include_continuous_features:
        continuous_features = np.zeros(3)
        return np.concatenate((index_encoded, continuous_features))
    return index_encoded

def prepare_brawler_data() -> Dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, BRAWLERS_JSON_PATH), 'r') as json_file:
        return json.load(json_file)


def get_brawler_vector(brawler_name: str, brawler_data: Dict, encoder: OneHotEncoder,
                       scaler: StandardScaler, include_continuous_features: bool) -> np.ndarray:
    if brawler_name == 'placeholder':
        return get_dummy_vector(brawler_data, encoder, scaler, include_continuous_features)

    brawler = brawler_data[brawler_name]
    index_encoded = encoder.transform([[brawler['index']]]).toarray()[0]

    if not include_continuous_features:
        return index_encoded

    continuous_features = [
        brawler['movement speed']['normal'],
        brawler['range']['normal'],
        brawler['level stats']['health']['11']
    ]
    continuous_features_scaled = scaler.transform([continuous_features])[0]
    return np.concatenate((index_encoded, continuous_features_scaled))


def get_match_vector(combination: List[str], match_dictionary: Dict, dummy_vector: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [match_dictionary[player] if player in combination else dummy_vector for player in all_players])

def get_match_vector_without_dummy(combination: List[str], match_dictionary: Dict) -> np.ndarray:
    return np.concatenate(
        [match_dictionary[player] for player in combination])

def get_match_dictionary(match: pd.Series, brawler_data: Dict, scaler: StandardScaler,
                          encoder: OneHotEncoder, include_continuous_features: bool = False) -> Dict:
    return {f'{team}{i}': get_brawler_vector(str.lower(match[f'{team}{i}']), brawler_data, encoder, scaler,
                                         include_continuous_features=include_continuous_features)
        for team in ['a', 'b'] for i in range(1, 4)}

def create_brawler_matrix(match_data: pd.DataFrame, brawler_data: Dict, scaler: StandardScaler,
                          encoder: OneHotEncoder, limit: int = None, include_phases: bool = False,
                          include_continuous_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    results = []

    if limit is not None:
        match_data = match_data.head(limit)

    for _, row in match_data.iterrows():
        match_dictionary = get_match_dictionary(row, brawler_data, scaler, encoder, include_continuous_features)
        dummy_vector = np.zeros(next(iter(match_dictionary.values())).shape)

        if include_phases:
            for combination in picking_combinations1 + picking_combinations2:
                vectors.append(get_match_vector(combination, match_dictionary, dummy_vector))
                results.append(row['result'])
        else:
            vectors.append(np.concatenate([match_dictionary[player] for player in all_players]))
            results.append(row['result'])

    return np.array(vectors), np.array(results)

def create_brawler_matrix_without_dummies(match_data: pd.DataFrame, brawler_data: Dict, scaler: StandardScaler,
                                        encoder: OneHotEncoder, limit: int = None,
                                        include_continuous_features: bool = False) -> Tuple[dict, np.ndarray]:
    vector_dict = {}
    results = []

    for combination in picking_combinations1_new + picking_combinations2_new:
        vector_dict[str(combination)] = []

    if limit is not None:
        match_data = match_data.head(limit)

    for _, row in match_data.iterrows():
        match_dictionary = get_match_dictionary(row, brawler_data, scaler, encoder, include_continuous_features)
        for combination in picking_combinations1_new + picking_combinations2_new:
            vector_dict[str(combination)].append(get_match_vector_without_dummy(combination, match_dictionary))
        results.append(row['result'])

    return vector_dict, np.array(results)

def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 64,
                learning_rate: float = 0.0005, shape_path: str = SHAPE_PATH) -> BrawlStarsNN:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BrawlStarsNN(X.shape[1])

    here = os.path.dirname(os.path.abspath(__file__))
    dump(X.shape[1], os.path.join(here, shape_path))

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(torch.FloatTensor(X_train))
            train_loss = criterion(train_outputs, torch.FloatTensor(y_train).unsqueeze(1))
            train_accuracy = ((train_outputs > 0.5) == torch.FloatTensor(y_train).unsqueeze(1)).float().mean()

            test_outputs = model(torch.FloatTensor(X_test))
            test_loss = criterion(test_outputs, torch.FloatTensor(y_test).unsqueeze(1))
            test_accuracy = ((test_outputs > 0.5) == torch.FloatTensor(y_test).unsqueeze(1)).float().mean()

        scheduler.step(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print("---")

    return model


def get_next_pick_combination(partial_comp: Dict, first_pick: bool) -> List[str]:
    picking_combination = picking_combinations1 if first_pick else picking_combinations2
    next_pick_combination = picking_combination[len(partial_comp)] if partial_comp else picking_combination[0]
    print(next_pick_combination)
    return next_pick_combination

def predict_best_pick(model: BrawlStarsNN, partial_comp: Dict, brawler_data: Dict, next_pick_combination: List[str],
                      encoder: OneHotEncoder, scaler: StandardScaler, device: torch.device,
                      include_dummies: bool, include_continuous_features: bool) -> List[Tuple[str, float]]:

    match_dictionary = {k: get_brawler_vector(v, brawler_data, encoder, scaler,
                                              include_continuous_features=include_continuous_features)
                        for k, v in partial_comp.items()}
    dummy_vector = np.zeros(next(iter(match_dictionary.values())).shape)

    brawler_win_rates = {}
    model.eval()
    with torch.no_grad():
        for brawler_name in brawler_data.keys():
            if brawler_name not in partial_comp.values():
                next_pick = next(player for player in next_pick_combination if player not in partial_comp)
                match_dictionary[next_pick] = get_brawler_vector(brawler_name, brawler_data, encoder, scaler,
                                                                 include_continuous_features=include_continuous_features)
                if include_dummies:
                    match_vector = get_match_vector(next_pick_combination, match_dictionary, dummy_vector)
                else:
                    match_vector = get_match_vector_without_dummy(next_pick_combination, match_dictionary)
                input_tensor = torch.FloatTensor(match_vector).unsqueeze(0).to(device)
                win_rate = torch.sigmoid(model(input_tensor)).item()
                brawler_win_rates[brawler_name] = win_rate

    return sorted(brawler_win_rates.items(), key=lambda x: x[1], reverse=True)

def train_without_dummies(match_data: pd.DataFrame, brawler_data: dict, scaler: StandardScaler, encoder: OneHotEncoder,
                          map: str, include_continuous_features: bool):
    X_dict, y = create_brawler_matrix_without_dummies(match_data, brawler_data, scaler, encoder,
                                 include_continuous_features=include_continuous_features)

    for combination in picking_combinations1_new + picking_combinations2_new:
        X = np.array(X_dict[str(combination)])
        map_base_path = f'out/models/{map.replace(" ", "_").lower()}'
        if not os.path.exists(f'{map_base_path}'):
            os.makedirs(f'{map_base_path}')
        shape_path = f'{map_base_path}/{str(combination)}_shape.joblib'
        model = train_model(X, y, shape_path=shape_path)
        torch.save(model.state_dict(), f'{map_base_path}/{str(combination)}.pth')

def train_with_dummies(match_data: pd.DataFrame, brawler_data: dict, scaler: StandardScaler, encoder: OneHotEncoder,
                          map: str, include_continuous_features: bool):
    X, y = create_brawler_matrix(match_data, brawler_data, scaler, encoder, include_phases=True,
                                 include_continuous_features=include_continuous_features)

    model = train_model(X, y)

    torch.save(model.state_dict(), f'out/models/{map.replace(" ", "_").lower()}.pth')

def train_map(map: str, include_continuous_features: bool, include_dummies: bool):
    match_data = prepare_training_data(map=map)
    brawler_data = prepare_brawler_data()

    unique_indices = np.array([[brawler['index']] for brawler in brawler_data.values()])
    encoder = OneHotEncoder()
    encoder.fit(unique_indices)

    continuous_features = [
        [brawler['movement speed']['normal'], brawler['range']['normal'], brawler['level stats']['health']['11']]
        for brawler in brawler_data.values()
    ]
    scaler = StandardScaler()
    scaler.fit(continuous_features)

    here = os.path.dirname(os.path.abspath(__file__))
    dump(encoder, os.path.join(here, ENCODER_PATH))
    dump(scaler, os.path.join(here, SCALER_PATH))

    if include_dummies:
        train_with_dummies(match_data, brawler_data, scaler, encoder, map, include_continuous_features)
    else:
        train_without_dummies(match_data, brawler_data, scaler, encoder, map, include_continuous_features)

def training_cycle(include_continuous_features: bool, include_dummies: bool):
    db = Database()
    all_maps = db.getAllMaps()
    all_maps = [x[0] for x in all_maps]
    for map in all_maps:
        print(f"Training model for {map}")
        train_map(map, include_continuous_features, include_dummies)


if __name__ == '__main__':
    train_map("Out in the Open", include_continuous_features=False, include_dummies=False)
