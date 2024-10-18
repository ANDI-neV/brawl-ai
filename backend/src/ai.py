import json
import os
import string
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine, URL
import configparser
import requests

BRAWLERS_JSON_PATH = 'out/brawlers/brawlers.json'
BRAWLER_WINRATES_JSON_PATH = 'out/brawlers/brawler_winrates.json'
BRAWLER_PICKRATES_JSON_PATH = 'out/brawlers/brawler_pickrates.json'
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


def get_map_winrate(map):
    here = os.path.dirname(os.path.abspath(__file__))
    with (open(os.path.join(here, BRAWLER_WINRATES_JSON_PATH), 'r')
          as json_file):
        return json.load(json_file)[map]


def get_map_pickrate(map):
    here = os.path.dirname(os.path.abspath(__file__))
    with (open(os.path.join(here, BRAWLER_PICKRATES_JSON_PATH), 'r')
          as json_file):
        return json.load(json_file)[map]


def get_filtered_brawlers(player_tag, min_level):
    url = "https://api.brawlstars.com"
    config = configparser.ConfigParser()
    config.read('config.ini')
    token = config["Credentials"]["api"]
    headers = {
        "Authorization": f"Bearer {token}"
    }

    if player_tag.startswith('#'):
        player_tag = player_tag[1:]

    try:
        response = requests.get(f"{url}/v1/players/%23{player_tag}", headers=headers)
        response.raise_for_status()

        data = response.json()
        print(f"API Response: {data}")

        brawlers = data.get("brawlers", [])
        filtered_brawlers = [
            str.lower(brawler["name"])
            for brawler in brawlers
            if brawler.get("power", 0) >= min_level
        ]

        return filtered_brawlers

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Brawl Stars API: {e}")
        return []


brawler_data = prepare_brawler_data()
n_brawlers = len(brawler_data)
CLS_TOKEN_INDEX = n_brawlers
PAD_TOKEN_INDEX = n_brawlers + 1
n_brawlers_with_special_tokens = n_brawlers + 2


def prepare_training_data() -> pd.DataFrame:
    """
    Loads training data from database in form of a pandas Dataframe.

    Returns:
        pd.Dataframe: Dataframe that contains training data.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    url_object = URL.create(
        "postgresql+psycopg2",
        username=config['Credentials']['username'],
        password=config['Credentials']['password'],
        host=config['Credentials']['host'],
        database=config['Credentials']['database'],
    )
    engine = create_engine(url_object)
    query = "SELECT * FROM battles"
    match_data = pd.read_sql_query(query, con=engine)

    return match_data


class BrawlStarsTransformer(nn.Module):
    """
    A Transformer-based neural network architecture for sequential predictions
    of Brawl Stars brawler combinations.

    This model utilizes a Transformer encoder to process sequences of brawlers,
    the associated team, positions, and map information to output a pick-score
    for all brawlers in the game in terms of a score from 0 to 1, where the sum
    of all scores is 1. This allows the prediction of the next best brawlers
    for selection.

    Attributes:
        brawler_embedding (nn.Embedding): Embedding layer for brawler tokens.
        map_embedding (nn.Embedding): Embedding layer for map identifiers.
        team_projection (nn.Linear): Linear projection for team indicators.
        position_embedding (nn.Embedding): Embedding layer for position
            information.
        transformer_encoder (nn.TransformerEncoder): Main Transformer encoder.
        output_layer (nn.Linear): Final linear layer for brawler prediction.

    Args:
        n_brawlers (int): Number of unique brawlers in the game.
        n_maps (int): Number of unique maps in the game.
        d_model (int, optional): Dimension of the model.
            Defaults to 64.
        nhead (int, optional): Number of heads in multi-head attention.
            Defaults to 4.
        num_layers (int, optional): Number of sub-encoder-layers in the
            encoder. Defaults to 2.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 7.

    Note:
        The model uses special tokens: PAD_TOKEN_INDEX for padding and assumes
        a CLS token at position 0 (Thus the sequence length of 7 instead of 6).
    """
    def __init__(self, n_brawlers, n_maps, d_model=64, nhead=4, num_layers=2,
                 dropout=0.1, max_seq_len=7):

        super().__init__()
        self.brawler_embedding = nn.Embedding(
            n_brawlers_with_special_tokens, d_model,
            padding_idx=PAD_TOKEN_INDEX
        )
        self.map_embedding = nn.Embedding(n_maps, d_model)
        self.team_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(
            max_seq_len, d_model, padding_idx=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, n_brawlers)
        print(f"Output layer size: {n_brawlers}")

    def forward(self, brawlers, team_indicators, positions, map_id,
                src_key_padding_mask=None):
        """
        Forward pass of the BrawlStarsTransformer.

        Args:
            brawlers (Tensor): Tensor of brawler indices.
            team_indicators (Tensor): Tensor indicating team affiliations.
            positions (Tensor): Tensor of position indices.
            map_id (Tensor): Tensor of map identifiers.
            src_key_padding_mask (Tensor, optional): Mask for padding tokens.
                Defaults to None.

        Returns:
            Tensor: Logits for brawler prediction.

        Note:
            The method combines brawler, team, position, and map embeddings
            before passing through the Transformer. It uses the CLS token
            output for final prediction.
        """
        x = self.brawler_embedding(brawlers)
        x = x + self.team_embedding(team_indicators)
        x = x + self.position_embedding(positions)
        x = x + self.map_embedding(map_id).unsqueeze(1)

        x = self.transformer_encoder(x,
                                     src_key_padding_mask=src_key_padding_mask)

        # Use the output corresponding to the CLS token for prediction
        cls_output = x[:, 0, :]  # CLS token is at position 0

        logits = self.output_layer(cls_output)
        return logits


def get_opposite_team(team: str) -> str:
    return 'a' if team == 'b' else 'b'


def get_match_dictionary(match: pd.Series, result: bool) -> Dict:
    """
    Creates a dictionary for a given match. Each match 6 players,
    with 3 on each team, with 'a' representing team 1 and 'b' team 2.
    The brawlers picked by the players are converted into their unique
    indices.

    Args:
        match (pd.Series): Match that was played.
        result (bool): Result of the match played.
    Returns:
        Dict: Converted Match.
    """
    if result:
        return {f'{team}{i}': get_brawler_index(match[f'{team}{i}'])
                for team in ['a', 'b'] for i in range(1, 4)}
    else:
        return {f'{team}{i}':
                get_brawler_index(match[f'{get_opposite_team(team)}{i}'])
                for team in ['a', 'b'] for i in range(1, 4)}


def get_match_vector(combination: List[str], match_dictionary: Dict) \
        -> np.ndarray:
    """
    Converts a combination of players into a vector representation using
    a match dictionary.

    Args:
        combination (List[str]): A list of player identifiers.
        match_dictionary (Dict): A dictionary mapping player identifiers
            to their vector representations.

    Returns:
        np.ndarray: A numpy array containing the vector representations
            of the players in the combination.

    Note:
        This function assumes that all players in the combination exist
        in the match_dictionary.
    """
    match_list = []
    for player in combination:
        match_list.append(match_dictionary[player])

    return np.array(match_list)


def get_brawler_vectors(match_data: pd.DataFrame,
                        map_id_mapping: Dict[str, int],
                        limit: int = None) -> List[Dict]:
    """
    Generates training samples from match data for brawler prediction.

    Args:
        match_data (pd.DataFrame): DataFrame containing match data.
        map_id_mapping (Dict[str, int]): Dictionary mapping map names
            to their integer IDs.
        limit (int, optional): Limit on the number of rows to process
            from match_data. Defaults to None.

    Returns:
        List[Dict]: A list of dictionaries, each representing a
            training sample with keys:
            - 'current_picks': List of current brawler picks.
            - 'team_indicators': List indicating team affiliation for
                    each pick.
            - 'positions': List of position indices for each pick.
            - 'next_pick': The next brawler pick.
            - 'map_id': Integer ID of the map.
    """

    training_samples = []
    invalid_samples = 0

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

            if any(pick < 1 or pick > PAD_TOKEN_INDEX for pick in
                   current_picks) or next_pick < 1 or next_pick > PAD_TOKEN_INDEX:
                invalid_samples += 1
                continue

            team_indicators = [1 if player[0] == 'a' else 2 for player in combination[:-1]]
            positions = [i for i in range(len(current_picks))]

            training_samples.append({
                'current_picks': current_picks,
                'team_indicators': team_indicators,
                'positions': positions,
                'next_pick': next_pick,
                'map_id': map_id,
            })

    print(f"Total training samples: {len(training_samples)}")
    print(f"Invalid samples skipped: {invalid_samples}")
    return training_samples

def get_brawler_dict(picks: List[str], first_pick: bool) -> Dict[str, str]:
    """
        Creates a dictionary mapping pick positions to brawlers based
        on the pick order.

        Args:
            picks (List[str]): List of brawler picks in order.
            first_pick (bool): Boolean indicating if this is the first
                pick team.

        Returns:
            Dict[str, str]: A dictionary mapping pick positions
                (e.g., 'a1', 'b2') to brawler names.

        """
    brawler_dict = {}
    if first_pick:
        for i, brawler in enumerate(picks):
            brawler_dict[first_pick_sequence[i]] = picks[i]
            print(f"brawler dict (first): {brawler_dict}")
    else:
        for i, brawler in enumerate(picks):
            brawler_dict[second_pick_sequence[i]] = picks[i]
            print(f"brawler dict (second): {brawler_dict}")
    return brawler_dict


def predict_next_pick(model, current_picks, team_indicators, map_id):
    """
   Predicts the next brawler pick using the trained model.

   Args:
       model: The trained BrawlStarsTransformer model.
       current_picks (List[int]): List of current brawler picks
        (as integer indices).
       team_indicators (List[int]): List indicating team affiliation
        for each pick.
       map_id (int): Integer ID of the map.

   Returns:
       Tuple[int, torch.Tensor]: A tuple containing:
           - The predicted next pick (as an integer index).
           - A tensor of probabilities for all possible next picks.

   Note:
       This function handles the conversion of inputs to tensors and
       moves them to the appropriate device. It uses torch.no_grad()
       for inference to disable gradient computation.
   """
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
    index = brawler_data.get(str.lower(brawler), {}).get("index")
    if index is None:
        print(f"Warning: Brawler '{brawler}' not found in brawler_data")
        return None
    return index


def train_transformer_model(training_samples, n_brawlers, n_maps, d_model=64,
                            nhead=4, num_layers=2, batch_size=64, epochs=10,
                            learning_rate=0.001):
    """
    Trains the BrawlStarsTransformer model on the provided training samples.

    Args:
        training_samples (List[Dict]): List of training samples, each a
            dictionary with pick data.
        n_brawlers (int): Number of unique brawlers in the game.
        n_maps (int): Number of unique maps in the game.
        d_model (int, optional): Dimension of the model. Defaults to 64.
        nhead (int, optional): Number of heads in multi-head attention.
            Defaults to 4.
        num_layers (int, optional): Number of sub-encoder-layers in the
            encoder. Defaults to 2.
        batch_size (int, optional): Size of each training batch.
            Defaults to 64.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer.
            Defaults to 0.001.

    Returns:
        BrawlStarsTransformer: The trained model.

    Note:
        This function initializes the model, moves it to the available device
        (CPU or CUDA), and trains it using CrossEntropyLoss and Adam optimizer.
        It prints the loss for each epoch.
    """
    model = BrawlStarsTransformer(PAD_TOKEN_INDEX + 1, n_maps, d_model,
                                  nhead, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(training_samples), batch_size):
            batch = prepare_batch(training_samples[i:i + batch_size])

            # Adjust validation
            if torch.max(batch['target']) > PAD_TOKEN_INDEX or torch.min(batch['target']) < 1:
                print(
                    f"Invalid target values found. Min: {torch.min(batch['target'])}, Max: {torch.max(batch['target'])}")
                continue

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

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Loss: {total_loss / len(training_samples):.4f}")

    return model


def prepare_batch(samples, max_seq_len=7):
    """
    Prepares a batch of samples for input into the BrawlStarsTransformer model.

    Args:
        samples (List[Dict]): List of sample dictionaries, each containing
            pick data.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 7.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - 'brawlers': Tensor of brawler indices, including CLS token
                    and padding.
            - 'team_indicators': Tensor of team indicators, including for CLS
                    token and padding.
            - 'positions': Tensor of position indices, adjusted for CLS token
                    and padding.
            - 'map_id': Tensor of map IDs.
            - 'target': Tensor of target (next pick) indices.
            - 'padding_mask': Boolean tensor indicating padded positions.

    Note:
        This function prepends a CLS token to each sequence, pads sequences to
        max_seq_len and prepares all required inputs for the transformer model.
    """
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
        team_indicators_padded = team_indicators + [0] * padding_length
        positions_padded = positions + [0] * padding_length

        padding_mask = [False] * seq_len + [True] * padding_length

        brawlers_list.append(brawlers_padded)
        team_indicators_list.append(team_indicators_padded)
        positions_list.append(positions_padded)
        target_list.append(target)
        map_ids.append(map_id)
        padding_masks.append(padding_mask)

    return {
        'brawlers': torch.tensor(brawlers_list, dtype=torch.long),
        'team_indicators': torch.tensor(team_indicators_list,
                                        dtype=torch.long),
        'positions': torch.tensor(positions_list, dtype=torch.long),
        'map_id': torch.tensor(map_ids, dtype=torch.long),
        'target': torch.tensor(target_list, dtype=torch.long),
        'padding_mask': torch.tensor(padding_masks, dtype=torch.bool)
    }

def create_map_id_mapping(match_data: pd.DataFrame) -> Dict[str, int]:
    """
    Creates a mapping from map names to unique integer IDs.

    Args:
        match_data (pd.DataFrame): DataFrame containing match data with a
        'map' column.

    Returns:
        Dict[str, int]: A dictionary mapping each unique map name to a
        unique integer ID.
    """
    unique_maps = match_data['map'].unique()
    return {map_name: idx for idx, map_name in enumerate(unique_maps)}


index_to_brawler_name = {data['index']: name for name, data
                         in brawler_data.items()}


def prepare_input(current_picks_dict, map_name, map_id_mapping,
                  first_pick, max_seq_len=7):
    """
    Prepares input data for the BrawlStarsTransformer model based on current
    picks and map.

    Args:
        current_picks_dict (Dict[str, str]): Dictionary of current picks,
            mapping positions to brawler names.
        map_name (str): Name of the current map.
        map_id_mapping (Dict[str, int]): Mapping of map names to their
            corresponding IDs.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 7.
        first_pick (bool): Team that has pick priority.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing model input tensors:
            - 'brawlers': Tensor of brawler indices, including CLS token
                    and padding.
            - 'team_indicators': Tensor of team indicators.
            - 'positions': Tensor of position indices.
            - 'map_id': Tensor of the map ID.
            - 'padding_mask': Boolean tensor for padding positions.

    Raises:
        ValueError: If no matching combination is found for the provided
        positions or if the map is not found.
    """
    # Possible picking combinations used during training
    print(f"first_pick: {first_pick}")
    if (first_pick):
        possible_combinations = picking_combinations1
    else:
        possible_combinations = picking_combinations2
    print(f"possible combos: {possible_combinations}")
    # Find a matching combination
    print(f"Current picks: {current_picks_dict}")
    for combination in possible_combinations:
        if (all(pos in current_picks_dict for pos in combination[:-1]) and
                len(combination[:-1]) == len(current_picks_dict)):
            selected_combination = combination
            print(f"Selected combination: {selected_combination}")
            break
    else:
        raise ValueError("No matching combination found for "
                         "the provided positions.")

    # Get current picks and positions
    current_picks_names = [current_picks_dict[pos] for pos
                           in selected_combination[:-1]]
    current_picks_positions = selected_combination[:-1]
    current_picks_indices = [get_brawler_index(brawler_name)
                             for brawler_name in current_picks_names]

    # Prepare model inputs
    brawlers = [CLS_TOKEN_INDEX] + current_picks_indices
    team_indicators = [0] + [1 if pos[0] == 'a' else 2 for pos
                             in current_picks_positions]
    positions = [0] + [i + 1 for i in range(len(current_picks_indices))]

    seq_len = len(brawlers)
    padding_length = max_seq_len - seq_len
    brawlers_padded = brawlers + [PAD_TOKEN_INDEX] * padding_length
    team_indicators_padded = team_indicators + [0] * padding_length
    positions_padded = positions + [0] * padding_length
    padding_mask = [False] * seq_len + [True] * padding_length

    # Convert to tensors
    brawlers_tensor = torch.tensor([brawlers_padded], dtype=torch.long)
    team_indicators_tensor = torch.tensor([team_indicators_padded],
                                          dtype=torch.long)
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
    """
    Predicts the next brawler pick using the trained model.

    Args:
       model (BrawlStarsTransformer): The trained transformer model.
       input_data (Dict[str, torch.Tensor]): Input data as prepare
            by prepare_input function.
       already_picked_indices (List[int]): Indices of brawlers that
            have already been picked.

   Returns:
       Tuple[int, torch.Tensor]: A tuple containing:
           - The index of the predicted next brawler pick.
           - A tensor of probabilities for all possible next picks.

   Note:
       This function sets the probabilities of already picked brawlers
       to negative infinity before making the prediction.
   """
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


def test_team_composition(model, current_picks_dict, map_name,
                          map_id_mapping, first_pick, max_seq_len=7):
    """
    Tests a team composition by predicting the next brawler pick and
    calculating probabilities for all brawlers.

    Args:
        model (BrawlStarsTransformer): The trained transformer model.
        current_picks_dict (Dict[str, str]): Dictionary of current picks,
            mapping positions to brawler names.
        map_name (str): Name of the current map.
        map_id_mapping (Dict[str, int]): Mapping of map names to their
            corresponding IDs.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 7.
        first_pick (bool): Team that has pick priority.

    Returns:
        Dict[str, float]: A dictionary mapping brawler names to their
        predicted probabilities.

    Note:
        This function prints the predicted next pick and probabilities
        for all brawlers. It uses global variables get_brawler_index and
        index_to_brawler_name.
    """
    input_data = prepare_input(current_picks_dict, map_name,
                               map_id_mapping, max_seq_len,
                               first_pick)

    already_picked_brawlers = [get_brawler_index(brawler_name)
                               for brawler_name in current_picks_dict.values()]

    predicted_brawler_index, probabilities = (
        predict_next_brawler(model, input_data, already_picked_brawlers))
    predicted_brawler_name = index_to_brawler_name.get(predicted_brawler_index,
                                                       'Unknown Brawler')
    print(f"Predicted next pick: {predicted_brawler_name}")

    probabilities[0, already_picked_brawlers] = 0

    probability_dict = {}

    print("\nProbabilities for all brawlers:")
    for idx in range(probabilities.size(1)):
        brawler_name = index_to_brawler_name.get(idx, 'Unknown Brawler')
        prob = probabilities[0, idx].item()
        print(f"{brawler_name}: {prob:.4f}")
        probability_dict[brawler_name] = prob

    return probability_dict


def get_all_maps():
    map_json = 'out/brawlers/map_data.json'
    with open(map_json, 'r') as f:
        map_id_mapping = json.load(f)

    current_maps_json = 'out/models/map_id_mapping.json'
    with open(current_maps_json, 'r') as f:
        current_maps = json.load(f)

    filtered_maps = {}
    for active_map in list(current_maps.keys()):
        filtered_maps[active_map] = map_id_mapping[string.capwords(active_map).replace("'", "")]

    print(filtered_maps)
    return filtered_maps


def train_model():
    """
    Trains the BrawlStarsTransformer model using prepared match data.

    This function performs the following steps:
    1. Prepares training data from match data.
    2. Prints information about the brawlers and special tokens.
    3. Creates a map ID mapping and saves it to a JSON file.
    4. Generates training samples from the match data.
    5. Trains the transformer model.
    6. Saves the trained model to a file.
    """
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
    training_samples = get_brawler_vectors(match_data,
                                           map_id_mapping=map_id_mapping,
                                           limit=1000)

    print("Map ID Mapping:")
    for map_name, map_id in map_id_mapping.items():
        print(f"{map_name}: {map_id}")

    with open(os.path.join(here, 'out/models/map_id_mapping.json'), 'w') as f:
        json.dump(map_id_mapping, f)

    model = train_transformer_model(training_samples, n_brawlers, n_maps)
    torch.save(model.state_dict(), 'out/models/transformer_3.pth')


def load_model(n_brawlers, n_maps, model_path='out/models/transformer_3.pth',
               d_model=64, nhead=4, num_layers=2):
    """
    Loads a trained BrawlStarsTransformer model from a file.

    Args:
        n_brawlers (int): Number of unique brawlers in the game.
        n_maps (int): Number of unique maps in the game.
        model_path (str, optional): Path to the saved model file.
            Defaults to 'out/models/transformer.pth'.
        d_model (int, optional): Dimension of the model. Defaults to 64.
        nhead (int, optional): Number of heads in multi-head attention.
            Defaults to 4.
        num_layers (int, optional): Number of sub-encoder-layers in the encoder.
            Defaults to 2.

    Returns:
        BrawlStarsTransformer: The loaded model, moved to the appropriate
        device (CPU or CUDA).

    Note:
        This function initializes a new BrawlStarsTransformer model with the
        given parameters, loads the state dict from the specified file, and
        moves the model to the available device.
    """
    model = BrawlStarsTransformer(n_brawlers_with_special_tokens, n_maps, d_model, nhead,
                                  num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def predict(picks_dict, map_name, first_pick):
    """
    Predicts the next brawler pick based on the current picks and map.

    Args:
        picks_dict (Dict[str, str]): A dictionary of current picks, mapping
            positions to brawler names.
        map_name (str): The name of the current map.
        first_pick (bool): Team that has pick priority.

    Returns:
        Dict[str, float]: A dictionary mapping brawler names to their
            predicted probabilities.

    Note:
        This function prepares the necessary data, loads the trained model,
        and uses the test_team_composition function to make predictions.
    """
    brawler_data = prepare_brawler_data()
    n_brawlers = len(brawler_data)
    map_id_mapping = load_map_id_mapping()
    n_maps = len(map_id_mapping)
    model = load_model(n_brawlers, n_maps)

    probability_dict = test_team_composition(model, picks_dict,
                                             map_name, map_id_mapping,
                                             first_pick)

    return probability_dict


def test():
    """
    Runs a test prediction with a predefined set of picks and map.

    This function demonstrates how to use the predict function with
    a sample scenario. It sets up a dictionary of current picks and
    a map name, then calls the predict function.

    Note:
        This is primarily for testing and demonstration purposes.
    """
    current_picks_dict = {
        'b1': 'buzz',
        'b2': 'edgar',
        'a1': 'barley',
        'a2': 'hank',
        'a3': 'darryl',
    }
    map_name = 'Out in the Open'
    predict(current_picks_dict, map_name, first_pick=True)


if __name__ == '__main__':
    train_model()
