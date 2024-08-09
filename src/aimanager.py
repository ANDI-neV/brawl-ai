import torch
from ai import *
from torchsummary import summary
from joblib import load

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = load(os.path.join(here, 'out/models/encoder.joblib'))
scaler = load(os.path.join(here,'out/models/scaler.joblib'))

def get_brawler_lib(us, brawlers): # if true, a1,b1,b2,a2,a3,b3, else b1,a1,a2,b2,b3,a3
    print("Us is "  + str(us))
    combination = {}
    if us:
        combination = {
            'a1': brawlers[0],
            'b1': brawlers[1],
            'b2': brawlers[2],
            'a2': brawlers[3],
            'a3': brawlers[4],
            'b3': brawlers[5]
        }
    else:
        combination = {
            'b1': brawlers[0],
            'a1': brawlers[1],
            'a2': brawlers[2],
            'b2': brawlers[3],
            'b3': brawlers[4],
            'a3': brawlers[5]
        }
    return combination

def get_map_path(map):
    mapPath = map.replace(" ", "_").lower()
    mapPath = 'out/models/brawl_stars_model_' + mapPath + '.pth'
    return mapPath

def get_finished_winrate(us, brawlers, map):
    brawler_data = prepare_brawler_data()
    #combination = get_brawler_lib(us, brawlers)
    combination = brawlers
    mapPath = get_map_path(map)

    input_data = prepare_input_data(combination, brawler_data, encoder, scaler)

    model = BrawlStarsNN(input_data.shape[1]).to(device)
    
    model.load_state_dict(torch.load(os.path.join(here, mapPath)))
    value = predict_win_probability(model, input_data, device)
    return value

def get_next_pick(us, brawlers, map):
    brawler_data = prepare_brawler_data()
    #combination = get_brawler_lib(us, brawlers)
    combination = brawlers
    print(combination) 
    print(type(combination))
    keys = list(combination.keys())
    for key in keys: #check if key = value, then delete if true
        if key == combination[key]:
            del combination[key]
    
    dummy_combination = {
        'a1': 'griff',
        'a2': 'rico',
        'a3': 'chester',
        'b1': 'surge',
        'b2': 'pam',
        'b3': 'penny'
    }

    mapPath = get_map_path(map)
    input_data = prepare_input_data(dummy_combination, brawler_data, encoder, scaler)

    model = BrawlStarsNN(input_data.shape[1]).to(device)
    model.load_state_dict(torch.load(os.path.join(here, mapPath)))

    best_picks = predict_best_pick(model, combination, brawler_data, encoder, scaler, device, first_pick=us)
    return best_picks
