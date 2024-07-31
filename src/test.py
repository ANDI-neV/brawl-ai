from db import amongus
from src.ai import *

brawler_data = prepare_brawler_data()

combination = {
    'a1': 'angelo',
    'a2': 'brock',
    'a3': 'piper',
    'b1': 'pam',
    'b2': 'mandy',
    'b3': 'hank'
}

input_data = prepare_input_data(combination, brawler_data)

scaler = StandardScaler()
scaler.fit(input_data)
normalized_data = normalize_input_data(input_data, scaler)
print(normalized_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size=normalized_data.shape[1]).to(device)
model.load_state_dict(torch.load('best_model.pth'))


# Predict win probability
win_probability = predict_win_probability(model, normalized_data, device)
print(f'Win Probability: {win_probability:.4f}')
