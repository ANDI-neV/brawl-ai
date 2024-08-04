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
model.load_state_dict(torch.load('brawl_stars_model_undermine2.pth'))
model.eval()

partial_comp = {'b1': 'shelly'}
best_picks = predict_best_pick(model, partial_comp, brawler_data, encoder, scaler, device, first_pick=False)
print("Top 5 recommended next picks based on predicted win rate:")
for brawler, win_rate in best_picks[:5]:
    print(f"{brawler}: {win_rate:.4f}")