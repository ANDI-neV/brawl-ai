import torch
from ai import *
from torchsummary import summary
from joblib import load

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = load(os.path.join(here, 'out/models/encoder.joblib'))
scaler = load(os.path.join(here,'out/models/scaler.joblib'))

def get_finished_winrate(brawlers, map): # brawlers is a list a1, a2, a3, b1, b2, b3
    brawler_data = prepare_brawler_data()
    combination = {
        'a1': brawlers[0],
        'a2': brawlers[1],
        'a3': brawlers[2],
        'b1': brawlers[3],
        'b2': brawlers[4],
        'b3': brawlers[5]
    }
    mapPath = map.replace(" ", "_").lower()

    input_data = prepare_input_data(combination, brawler_data, encoder, scaler)

    model = BrawlStarsNN(input_data.shape[1]).to(device)
    model.load_state_dict(torch.load(os.path.join(here,'out/models/brawl_stars_model_', mapPath, '.pth')))
    return predict_win_probability(model, input_data, device)
