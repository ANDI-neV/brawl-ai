import torch
from src.ai import *
from torchsummary import summary
from joblib import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# Example usage
brawler_data = prepare_brawler_data()

combination = {
    'a1': 'griff',
    'a2': 'rico',
    'a3': 'chester',
    'b1': 'surge',
    'b2': 'pam',
    'b3': 'penny'
}

input_data = prepare_input_data(combination, brawler_data, encoder, scaler)

model = BrawlStarsNN(input_data.shape[1]).to(device)
model.load_state_dict(torch.load('brawl_stars_model_undermine.pth'))
model.eval()

win_probability = predict_win_probability(model, input_data, device)
print(f'Win Probability: {win_probability:.4f}')