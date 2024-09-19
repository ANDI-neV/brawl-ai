import json
import os
from typing import Dict, List, Tuple
import math
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


BRAWLERS_JSON_PATH = 'out/brawlers/brawlers.json'
picking_combinations1 = [['a1', 'b1', 'b2', 'a2'],
                         ['a1', 'b1', 'b2', 'a2', 'a3']]
picking_combinations2 = [['b1', 'a1'],
                         ['b1', 'a1', 'a2'],
                         ['b1', 'a1', 'a2', 'b2', 'b3', 'a3']]
all_players = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']

first_pick_sequence = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
second_pick_sequence = ['b1', 'a1', 'a2', 'b2', 'b3', 'a3']


def prepare_brawler_data() -> Dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, BRAWLERS_JSON_PATH), 'r') as json_file:
        return json.load(json_file)

def get_all_brawlers():
    brawler_data = prepare_brawler_data()
    return list(brawler_data.keys())

def load_map_id_mapping():
    with open('out/models/map_id_mapping.json', 'r') as f:
        map_id_mapping = json.load(f)
        map_id_mapping = {str(k): v for k, v in map_id_mapping.items()}
    return map_id_mapping

brawler_data = prepare_brawler_data()
n_brawlers = len(brawler_data)
CLS_TOKEN_INDEX = n_brawlers
PAD_TOKEN_INDEX = n_brawlers + 1
n_brawlers_with_special_tokens = n_brawlers + 2

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


class BrawlStarsTransformer(nn.Module):
    def __init__(self, n_brawlers, n_maps, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq_len=7):
        super().__init__()
        self.brawler_embedding = nn.Embedding(
            n_brawlers_with_special_tokens, d_model, padding_idx=PAD_TOKEN_INDEX
        )
        self.map_embedding = nn.Embedding(n_maps, d_model)
        self.team_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(
            max_seq_len, d_model, padding_idx=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, n_brawlers)

    def forward(self, brawlers, team_indicators, positions, map_id, src_key_padding_mask=None):
        x = self.brawler_embedding(brawlers)
        team_feature = team_indicators.float().unsqueeze(-1)
        x = x + self.team_projection(team_feature)
        x = x + self.position_embedding(positions)
        x = x + self.map_embedding(map_id).unsqueeze(1)

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Use the output corresponding to the CLS token for prediction
        cls_output = x[:, 0, :]  # CLS token is at position 0

        logits = self.output_layer(cls_output)
        return logits


def get_opposite_team(team: str) -> str:
    return 'a' if team == 'b' else 'b'

def get_match_dictionary(match: pd.Series, result: bool) -> Dict:
    if result:
        return {f'{team}{i}': get_brawler_index(match[f'{team}{i}'])
            for team in ['a', 'b'] for i in range(1, 4)}
    else:
        return {f'{team}{i}': get_brawler_index(match[f'{get_opposite_team(team)}{i}'])
                for team in ['a', 'b'] for i in range(1, 4)}

def get_match_vector(combination: List[str], match_dictionary: Dict) -> np.ndarray:
    match_list = []
    for player in combination:
        match_list.append(match_dictionary[player])

    return np.array(match_list)


def get_brawler_vectors(match_data: pd.DataFrame, map_id_mapping: Dict[str, int], limit: int = None) -> List[Dict]:
    training_samples = []

    if limit is not None:
        match_data = match_data.head(limit)

    for _, row in match_data.iterrows():
        map_id = map_id_mapping[row['map']]
        result = row['result']
        match_dictionary = get_match_dictionary(row, result)

        for combination in picking_combinations1 + picking_combinations2:
            if any(pos not in match_dictionary for pos in combination):
                continue

            full_sequence = get_match_vector(combination, match_dictionary)
            current_picks = full_sequence[:-1]
            next_pick = full_sequence[-1]
            team_indicators = [1 if player[0] == 'a' else 2 for player in combination[:-1]]
            positions = [i for i in range(len(current_picks))]

            training_samples.append({
                'current_picks': current_picks,
                'team_indicators': team_indicators,
                'positions': positions,
                'next_pick': next_pick,
                'map_id': map_id,
            })

    return training_samples

def get_brawler_dict(picks: List[str], first_pick: bool) -> Dict[str, str]:
    brawler_dict = {}
    if first_pick:
        for i, player in enumerate(first_pick_sequence):
            brawler_dict[player] = picks[i]
    else:
        for i, player in enumerate(second_pick_sequence):
            brawler_dict[player] = picks[i]

    return brawler_dict

def predict_next_pick(model, current_picks, team_indicators, map_id):
    brawlers = torch.tensor([current_picks])
    team_indicators = torch.tensor([team_indicators])
    positions = torch.tensor([[i for i in range(len(current_picks))]])
    map_id = torch.tensor([map_id])

    device = next(model.parameters()).device
    brawlers = brawlers.to(device)
    team_indicators = team_indicators.to(device)
    positions = positions.to(device)
    map_id = map_id.to(device)

    with torch.no_grad():
        logits = model(brawlers, team_indicators, positions, map_id)
        probabilities = torch.softmax(logits, dim=-1)
        next_pick = torch.argmax(probabilities, dim=-1).item()

    return next_pick, probabilities.squeeze()


def get_brawler_index(brawler):
    return brawler_data.get(str.lower(brawler))["index"]


def train_transformer_model(training_samples, n_brawlers, n_maps, d_model=64, nhead=4, num_layers=2,
                            batch_size=64, epochs=100, learning_rate=0.001):
    model = BrawlStarsTransformer(n_brawlers, n_maps, d_model, nhead, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(training_samples), batch_size):
            batch = prepare_batch(training_samples[i:i + batch_size])

            brawlers = batch['brawlers'].to(device)
            team_indicators = batch['team_indicators'].to(device)
            positions = batch['positions'].to(device)
            map_id = batch['map_id'].to(device)
            target = batch['target'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            optimizer.zero_grad()
            logits = model(
                brawlers,
                team_indicators,
                positions,
                map_id,
                src_key_padding_mask=padding_mask
            )
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * brawlers.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_samples):.4f}")

    return model


def prepare_batch(samples, max_seq_len=7):
    brawlers_list = []
    team_indicators_list = []
    positions_list = []
    target_list = []
    map_ids = []
    padding_masks = []

    for s in samples:
        current_picks = s['current_picks']
        team_indicators = s['team_indicators']
        positions = s['positions']
        map_id = s['map_id']
        target = s['next_pick']

        # Prepend CLS token
        current_picks = [CLS_TOKEN_INDEX] + list(current_picks)
        team_indicators = [0] + list(team_indicators)  # 0 for CLS token
        positions = [0] + [p + 1 for p in positions]  # Shift positions by 1

        seq_len = len(current_picks)

        # Pad sequences
        padding_length = max_seq_len - seq_len
        brawlers_padded = current_picks + [PAD_TOKEN_INDEX] * padding_length
        team_indicators_padded = team_indicators + [0] * padding_length  # 0 for padding
        positions_padded = positions + [0] * padding_length  # 0 for padding positions

        padding_mask = [False] * seq_len + [True] * padding_length

        brawlers_list.append(brawlers_padded)
        team_indicators_list.append(team_indicators_padded)
        positions_list.append(positions_padded)
        target_list.append(target)
        map_ids.append(map_id)
        padding_masks.append(padding_mask)

    return {
        'brawlers': torch.tensor(brawlers_list, dtype=torch.long),
        'team_indicators': torch.tensor(team_indicators_list, dtype=torch.long),
        'positions': torch.tensor(positions_list, dtype=torch.long),
        'map_id': torch.tensor(map_ids, dtype=torch.long),
        'target': torch.tensor(target_list, dtype=torch.long),
        'padding_mask': torch.tensor(padding_masks, dtype=torch.bool)
    }

def create_map_id_mapping(match_data: pd.DataFrame) -> Dict[str, int]:
    unique_maps = match_data['map'].unique()
    return {map_name: idx for idx, map_name in enumerate(unique_maps)}

index_to_brawler_name = {data['index']: name for name, data in brawler_data.items()}


def prepare_input(current_picks_dict, map_name, map_id_mapping, max_seq_len=7):
    # Possible picking combinations used during training
    possible_combinations = picking_combinations1 + picking_combinations2
    # Find a matching combination
    for combination in possible_combinations:
        if all(pos in current_picks_dict for pos in combination[:-1]):
            selected_combination = combination
            print(f"Selected combination: {selected_combination}")
            break
    else:
        raise ValueError("No matching combination found for the provided positions.")

    # Get current picks and positions
    current_picks_names = [current_picks_dict[pos] for pos in selected_combination[:-1]]
    current_picks_positions = selected_combination[:-1]
    current_picks_indices = [get_brawler_index(brawler_name) for brawler_name in current_picks_names]

    # Prepare model inputs
    brawlers = [CLS_TOKEN_INDEX] + current_picks_indices
    team_indicators = [0] + [1 if pos[0] == 'a' else 2 for pos in current_picks_positions]
    positions = [0] + [i + 1 for i in range(len(current_picks_indices))]

    seq_len = len(brawlers)
    padding_length = max_seq_len - seq_len
    brawlers_padded = brawlers + [PAD_TOKEN_INDEX] * padding_length
    team_indicators_padded = team_indicators + [0] * padding_length
    positions_padded = positions + [0] * padding_length
    padding_mask = [False] * seq_len + [True] * padding_length

    # Convert to tensors
    brawlers_tensor = torch.tensor([brawlers_padded], dtype=torch.long)
    team_indicators_tensor = torch.tensor([team_indicators_padded], dtype=torch.long)
    positions_tensor = torch.tensor([positions_padded], dtype=torch.long)
    padding_mask_tensor = torch.tensor([padding_mask], dtype=torch.bool)
    map_id = map_id_mapping.get(map_name)
    if map_id is None:
        raise ValueError(f"Map '{map_name}' not found in map_id_mapping.")
    map_id_tensor = torch.tensor([map_id], dtype=torch.long)
    return {
        'brawlers': brawlers_tensor,
        'team_indicators': team_indicators_tensor,
        'positions': positions_tensor,
        'map_id': map_id_tensor,
        'padding_mask': padding_mask_tensor
    }


def predict_next_brawler(model, input_data, already_picked_indices):
    device = next(model.parameters()).device
    brawlers = input_data['brawlers'].to(device)
    team_indicators = input_data['team_indicators'].to(device)
    positions = input_data['positions'].to(device)
    map_id = input_data['map_id'].to(device)
    padding_mask = input_data['padding_mask'].to(device)
    with torch.no_grad():
        logits = model(
            brawlers,
            team_indicators,
            positions,
            map_id,
            src_key_padding_mask=padding_mask
        )
        logits[0, already_picked_indices] = -float('inf')
        probabilities = torch.softmax(logits, dim=-1)
        predicted_brawler_index = torch.argmax(probabilities, dim=-1).item()
    return predicted_brawler_index, probabilities


def test_team_composition(model, current_picks_dict, map_name, map_id_mapping, max_seq_len=7):
    input_data = prepare_input(current_picks_dict, map_name, map_id_mapping, max_seq_len)

    already_picked_brawlers = [get_brawler_index(brawler_name) for brawler_name in current_picks_dict.values()]

    predicted_brawler_index, probabilities = predict_next_brawler(model, input_data, already_picked_brawlers)
    predicted_brawler_name = index_to_brawler_name.get(predicted_brawler_index, 'Unknown Brawler')
    print(f"Predicted next pick: {predicted_brawler_name}")

    topk = 15
    probabilities[0, already_picked_brawlers] = 0
    topk_indices = torch.topk(probabilities, topk).indices.squeeze().tolist()
    print("\nTop predictions (excluding already picked brawlers):")
    for idx in topk_indices:
        brawler_name = index_to_brawler_name.get(idx, 'Unknown Brawler')
        prob = probabilities[0, idx].item()
        print(f"{brawler_name}: {prob:.4f}")


def train_model():
    match_data = prepare_training_data()

    n_brawlers = len(brawler_data)
    print(f"Number of brawlers: {n_brawlers}")
    print(f"Length of brawler_data: {len(brawler_data)}")
    print("Sample brawler_data entries:")
    for i, (key, value) in enumerate(brawler_data.items()):
        print(f"{key}: {value}")
        if i >= 5:
            break

    print(f"CLS_TOKEN_INDEX: {CLS_TOKEN_INDEX}")
    print(f"PAD_TOKEN_INDEX: {PAD_TOKEN_INDEX}")
    print(f"n_brawlers_with_special_tokens: {n_brawlers_with_special_tokens}")

    n_maps = len(match_data['map'].unique())
    map_id_mapping = create_map_id_mapping(match_data)
    here = os.path.dirname(os.path.abspath(__file__))
    training_samples = get_brawler_vectors(match_data, map_id_mapping = map_id_mapping, limit=50000)

    print("Map ID Mapping:")
    for map_name, map_id in map_id_mapping.items():
        print(f"{map_name}: {map_id}")

    with open(os.path.join(here, 'out/models/map_id_mapping.json'), 'w') as f:
        json.dump(map_id_mapping, f)

    model = train_transformer_model(training_samples, n_brawlers, n_maps)
    torch.save(model.state_dict(), 'out/models/transformer.pth')

def load_model(n_brawlers, n_maps, model_path='out/models/transformer.pth', d_model=64, nhead=4, num_layers=2):
    model = BrawlStarsTransformer(n_brawlers, n_maps, d_model, nhead, num_layers)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model
def predict(picks_dict, map_name):
    brawler_data = prepare_brawler_data()
    n_brawlers = len(brawler_data)
    map_id_mapping = load_map_id_mapping()
    n_maps = len(map_id_mapping)
    model = load_model(n_brawlers, n_maps)

    test_team_composition(model, picks_dict, map_name, map_id_mapping)

def test():
    current_picks_dict = {
        'b1': 'hank',
        'b2': 'edgar',
        'b3': 'rosa',
        'a1': 'brock',
        'a2': 'piper'
    }
    map_name = 'Out in the Open'
    predict(current_picks_dict, map_name)

if __name__ == '__main__':
    test()