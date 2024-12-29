import onnxruntime as ort
import numpy as np
from ai import (load_map_id_mapping, get_brawler_index, acquire_combination, initialize_brawler_data,
                get_brawler_class)
from threading import Lock
import torch

MODEL_PATH = "./out/models/model.onnx"
model_lock = Lock()
try:
    print("ONNX Runtime Version:", ort.__version__)
    ort_session = ort.InferenceSession("path/to/your/model.onnx")
    print("ONNX Runtime initialized successfully.")
except Exception as e:
    print("Error initializing ONNX Runtime:", e)


def reload_model():
    global ort_session
    with model_lock:
        print("Reloading ONNX model...")
        ort_session = ort.InferenceSession(MODEL_PATH)
        print("Model reloaded successfully.")


def prepare_input(brawler_dict, map_name,
                               map_id_mapping, first_pick, max_seq_len):
    selected_combination = acquire_combination(brawler_dict, first_pick)
    if not selected_combination:
        return None

    current_picks_names = [brawler_dict[pos] for pos in selected_combination[:-1]]
    current_picks_indices = [get_brawler_index(brawler_name) for brawler_name in current_picks_names]
    brawler_classes = [get_brawler_class(brawler_name) for brawler_name in current_picks_names]
    brawler_data, constants = initialize_brawler_data()

    brawlers = [constants['CLS_TOKEN_INDEX']] + current_picks_indices
    team_indicators = [1] + [1 if pos[0] == 'a' else 2 for pos in selected_combination[:-1]]
    positions = [0] + [i + 1 for i in range(len(current_picks_indices))]

    seq_len = len(brawlers)
    padding_length = max_seq_len - seq_len

    brawlers_padded = brawlers + [constants['PAD_TOKEN_INDEX']] * padding_length
    brawler_classes_padded = [constants['CLASS_CLS_TOKEN_INDEX']] + brawler_classes + [constants['CLASS_PAD_TOKEN_INDEX']] * padding_length
    team_indicators_padded = team_indicators + [0] * padding_length
    positions_padded = positions + [0] * padding_length
    padding_mask = [False] * seq_len + [True] * padding_length
    map_id = map_id_mapping.get(map_name)

    return {
        "brawlers": np.array([brawlers_padded], dtype=np.int64),
        "brawler_classes": np.array([brawler_classes_padded], dtype=np.int64),
        "team_indicators": np.array([team_indicators_padded], dtype=np.int64),
        "positions": np.array([positions_padded], dtype=np.int64),
        "map_id": np.array([map_id], dtype=np.int64),
        "src_key_padding_mask": np.array([padding_mask], dtype=np.bool_)
    }


def predict(brawler_dict, map_name, first_pick):
    map_id_mapping = load_map_id_mapping()
    input_data = prepare_input(brawler_dict, map_name, map_id_mapping, first_pick, max_seq_len=7)
    brawler_data, constants = initialize_brawler_data()
    if not input_data:
        return None

    with model_lock:
        outputs = ort_session.run(None, {
            "brawlers": input_data["brawlers"],
            "brawler_classes": input_data["brawler_classes"],
            "team_indicators": input_data["team_indicators"],
            "positions": input_data["positions"],
            "map_id": input_data["map_id"],
            "src_key_padding_mask": input_data["src_key_padding_mask"]
        })
    logits = outputs[0]
    logits = torch.tensor(logits)

    already_picked_indices = [get_brawler_index(brawler_name) for brawler_name in brawler_dict.values()]
    logits[0, already_picked_indices] = -float("inf")

    probabilities = torch.softmax(logits, dim=-1)
    probabilities[0, already_picked_indices] = 0

    probability_dict = {}
    for idx in range(probabilities.size(1)):
        brawler_name = constants['index_to_brawler_name'].get(idx, 'Unknown Brawler')
        prob = probabilities[0, idx].item()
        if brawler_name == 'Unknown Brawler':
            continue
        probability_dict[brawler_name] = prob

    return probability_dict


def test_example_composition():
    brawler_dict = {"a1": "shelly", "b1": "colt", "b2": "piper"}
    map_name = "Out in the Open"
    first_pick = True

    probabilities = predict(brawler_dict, map_name, first_pick)
    print("Probabilities:", probabilities)


if __name__ == '__main__':
    test_example_composition()
