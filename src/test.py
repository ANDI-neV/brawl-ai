import torch
from src.ai import *
from torchsummary import summary
from joblib import load

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = load(os.path.join(here, 'out/models/encoder.joblib'))
scaler = load(os.path.join(here,'out/models/scaler.joblib'))

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
model.load_state_dict(torch.load(os.path.join(here,'out/models/brawl_stars_model_out_in_the_open.pth')))
model.eval()

partial_comp = {'b1':'griff', 'a1': 'pam', 'a2': 'penny', 'b2': 'piper',  'b3': 'angelo'}
best_picks = predict_best_pick(model, partial_comp, brawler_data, encoder, scaler, device, first_pick=False)
print("Top 5 recommended next picks based on predicted win rate:")
for brawler, win_rate in best_picks[:78]:
    print(f"{brawler}: {win_rate:.4f}")