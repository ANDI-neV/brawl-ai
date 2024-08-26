import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from ai import *
import math


class BrawlStarsPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6):
        super().__init__()
        position = torch.tensor([[0], [1], [1], [0], [0], [1]])  # First pick order

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class BrawlStarsTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(d_model, dim_feedforward)
        self.pos_encoder = BrawlStarsPositionalEncoding(dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(dim_feedforward, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return torch.sigmoid(output)



class BrawlStarsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_sequence_data(match_data, brawler_data, encoder, scaler, include_continuous_features=True):
    sequences = []
    results = []

    for _, row in match_data.iterrows():
        sequence = []
        pick_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
        for position in pick_order:
            brawler_name = str.lower(row[position])
            brawler_vector = get_brawler_vector(brawler_name, brawler_data, encoder, scaler,
                                                include_continuous_features)
            sequence.append(brawler_vector)

        sequences.append(sequence)
        results.append(row['result'])

    return np.array(sequences), np.array(results)


def train_transformer_model(X, y, d_model, nhead, num_encoder_layers, dim_feedforward,
                            batch_size=32, epochs=20, learning_rate=0.001):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = BrawlStarsDataset(X_train, y_train)
    val_dataset = BrawlStarsDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BrawlStarsTransformer(d_model, nhead, num_encoder_layers, dim_feedforward)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X.transpose(0, 1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X.transpose(0, 1))
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return model


def predict_next_pick(model, partial_comp, brawler_data, encoder, scaler, device, include_continuous_features=True):
    model.eval()
    with torch.no_grad():
        brawler_win_rates = {}
        pick_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
        next_pick = next(pos for pos in pick_order if pos not in partial_comp)

        for brawler_name in brawler_data.keys():
            if brawler_name not in partial_comp.values():
                sequence = []
                for pos in pick_order:
                    if pos in partial_comp:
                        brawler = partial_comp[pos]
                    elif pos == next_pick:
                        brawler = brawler_name
                    else:
                        brawler = 'placeholder'

                    brawler_vector = get_brawler_vector(brawler, brawler_data, encoder, scaler, include_continuous_features)
                    sequence.append(brawler_vector)

                input_tensor = torch.FloatTensor(sequence).unsqueeze(1).to(device)
                win_rate = model(input_tensor).item()
                brawler_win_rates[brawler_name] = win_rate

    return sorted(brawler_win_rates.items(), key=lambda x: x[1], reverse=True)


def main():
    match_data = prepare_training_data(map="Out in the Open")
    brawler_data = prepare_brawler_data()

    unique_indices = np.array([[brawler['index']] for brawler in brawler_data.values()])
    encoder = OneHotEncoder()
    encoder.fit(unique_indices)

    continuous_features = [
        [brawler['movement speed']['normal'], brawler['range']['normal'], brawler['level stats']['health']['11']]
        for brawler in brawler_data.values()
    ]
    scaler = StandardScaler()
    scaler.fit(continuous_features)

    here = os.path.dirname(os.path.abspath(__file__))
    dump(encoder, os.path.join(here, ENCODER_PATH))
    dump(scaler, os.path.join(here, SCALER_PATH))

    X, y = prepare_sequence_data(match_data, brawler_data, encoder, scaler, include_continuous_features=False)

    dump(X.shape[2], os.path.join(here, SHAPE_PATH))

    d_model = X.shape[2]  # Feature dimension
    nhead = 4  # Number of attention heads
    num_encoder_layers = 2
    dim_feedforward = 128

    model = train_transformer_model(X, y, d_model, nhead, num_encoder_layers, dim_feedforward)

    partial_comp = {'a1': 'rosa', 'b1': 'angelo', 'b2': 'chester'}
    predictions = predict_next_pick(model, partial_comp, brawler_data, encoder, scaler, torch.device('cpu'))
    print("Predictions:", predictions[:5])

def make_prediction():
    partial_comp = {'a1': 'angelo', 'b1': 'pearl', 'b2': 'chester', 'a2': 'piper'}

    brawler_data = prepare_brawler_data()
    encoder = load(ENCODER_PATH)
    scaler = load(SCALER_PATH)
    shape = load(SHAPE_PATH)

    state_dict = torch.load('best_transformer_model.pth')

    d_model = state_dict['embedding.weight'].shape[1]
    nhead = 4  # Number of attention heads
    num_encoder_layers = 2
    dim_feedforward = 128

    model = BrawlStarsTransformer(d_model, nhead, num_encoder_layers, dim_feedforward)

    model.load_state_dict(state_dict)
    model.eval()

    include_continuous_features = d_model != len(encoder.categories_[0])

    predictions = predict_next_pick(model, partial_comp, brawler_data, encoder, scaler, torch.device('cpu'),
                                    include_continuous_features=include_continuous_features)
    print("Predictions:")
    for brawler, win_rate in predictions[:78]:
        print(f"{brawler}: {win_rate:.4f}")

if __name__ == '__main__':
    make_prediction()
