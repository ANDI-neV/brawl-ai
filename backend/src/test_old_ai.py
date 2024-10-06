from old_ai import *
from joblib import load
from db import Database


here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run_model(map: str, partial_comp: dict, first_pick: bool):
    encoder = load(os.path.join(here, 'out/models/encoder.joblib'))
    scaler = load(os.path.join(here,'out/models/scaler.joblib'))

    next_pick_combination = get_next_pick_combination(partial_comp, first_pick=first_pick)

    path = f'out/models/{map.replace(" ", "_").lower()}'
    shape = load(os.path.join(here, f'{path}/{str(next_pick_combination)}_shape.joblib'))

    brawler_data = prepare_brawler_data()

    model = BrawlStarsNN(shape).to(device)
    model.load_state_dict(torch.load(os.path.join(here,f'{path}/{str(next_pick_combination)}.pth')))
    model.eval()

    best_picks = predict_best_pick(model, partial_comp, brawler_data, next_pick_combination,encoder, scaler, device,
                                   include_dummies=False, include_continuous_features=False)
    print("Top recommended next picks based on predicted win rate:")
    for brawler, win_rate in best_picks[:78]:
        print(f"{brawler}: {win_rate:.4f}")

def check_brawler_significance(map = None):
    brawler_significance = {}
    db = Database()
    brawler_data = prepare_brawler_data()
    for brawler in brawler_data.keys():
        brawler_significance[brawler] = db.check_brawler_winrate_for_map(brawler = brawler.upper(), map_name=map)

    print("Brawler significance:")
    keys = list(brawler_significance.keys())
    values = list(brawler_significance.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

    for brawler, significance in sorted_dict.items():
        print(f"{brawler}: {significance}")

if __name__ == '__main__':
    partial_comp = {'a1': 'hank', 'b1': 'piper', 'b2': 'angelo', 'a2': 'barley'}
    run_model(map = "Out in the Open", partial_comp = partial_comp, first_pick=True)
