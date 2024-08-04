import torch
from src.ai import *
from torchsummary import summary
from joblib import load

brawler_data = prepare_brawler_data()

combination = {
    'a1': 'angelo',
    'a2': 'piper',
    'a3': 'brock',
    'b1': 'shelly',
    'b2': 'poco',
    'b3': 'sprout'
}

input_data = prepare_input_data(combination, brawler_data)

#scaler = load('std_scaler.bin')
normalized_data = input_data
#normalized_data = normalize_input_data(input_data, scaler)
print(normalized_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(normalized_data.shape[1]).to(device)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

win_probability = predict_win_probability(model, normalized_data, device)
print(f'Win Probability: {win_probability:.4f}')
