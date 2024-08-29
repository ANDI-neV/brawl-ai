import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import math
from ai import *

class BrawlerEmbedding(nn.Module):
    def __init__(self, num_brawlers, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_brawlers, embedding_dim)

    def forward(self, brawler_ids):
        return self.embedding(brawler_ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class BrawlStarsTransformer(nn.Module):
    def __init__(self, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.brawler_embedding = nn.Embedding(num_brawlers, embedding_dim)
        self.team_embedding = nn.Embedding(3, embedding_dim)  # 0: ally, 1: enemy, 2: empty
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=6)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, src, team_indicators):
        src = self.brawler_embedding(src) + self.team_embedding(team_indicators)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=0))  # Average over sequence length
        return torch.sigmoid(output)

class BrawlStarsDataset(Dataset):
    def __init__(self, X, team_indicators, y):
        self.X = torch.LongTensor(X)
        self.team_indicators = torch.LongTensor(team_indicators)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.team_indicators[idx], self.y[idx]

def debug_brawler_indices(X, brawler_data):
    unique_indices = np.unique(X)
    num_brawlers = len(brawler_data)
    print(f"Number of unique brawler indices: {len(unique_indices)}")
    print(f"Maximum brawler index: {np.max(unique_indices)}")
    print(f"Number of brawlers in brawler_data: {num_brawlers}")
    if np.max(unique_indices) >= num_brawlers:
        print("Warning: Maximum index is out of range for the number of brawlers")
        print("Indices causing issues:", unique_indices[unique_indices >= num_brawlers])

def validate_partial_comp(partial_comp):
    valid_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
    for i, key in enumerate(partial_comp.keys()):
        if key != valid_order[i]:
            return False
    return True

def prepare_sequence_data(match_data, brawler_data):
    sequences = []
    team_indicators = []
    results = []

    brawler_to_index = {name: idx for idx, name in enumerate(brawler_data.keys())}
    unknown_index = len(brawler_to_index)

    for _, row in match_data.iterrows():
        sequence = []
        indicators = []
        pick_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
        for position in pick_order:
            brawler_name = str.lower(row[position])
            brawler_index = brawler_to_index.get(brawler_name, unknown_index)
            sequence.append(brawler_index)
            indicators.append(0 if position.startswith('a') else 1)

        sequences.append(sequence)
        team_indicators.append(indicators)
        results.append(row['result'])

    return np.array(sequences), np.array(team_indicators), np.array(results)

def train_transformer_model(X, team_indicators, y, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward,
                            batch_size=32, epochs=50, learning_rate=0.001):
    X_train, X_val, team_indicators_train, team_indicators_val, y_train, y_val = train_test_split(
        X, team_indicators, y, test_size=0.2, random_state=42)

    train_dataset = BrawlStarsDataset(X_train, team_indicators_train, y_train)
    val_dataset = BrawlStarsDataset(X_val, team_indicators_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BrawlStarsTransformer(num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_team_indicators, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_team_indicators = batch_team_indicators.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X.transpose(0, 1), batch_team_indicators.transpose(0, 1))
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_team_indicators, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_team_indicators = batch_team_indicators.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X.transpose(0, 1), batch_team_indicators.transpose(0, 1))
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return model

def predict_next_pick(model, partial_comp, brawler_data, device):
    model.eval()
    with torch.no_grad():
        brawler_win_rates = {}
        pick_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
        next_pick = next(pos for pos in pick_order if pos not in partial_comp)

        brawler_to_index = {name: idx for idx, name in enumerate(brawler_data.keys())}
        unknown_index = len(brawler_to_index)

        for brawler_name in brawler_data.keys():
            if brawler_name not in partial_comp.values():
                sequence = []
                indicators = []
                for pos in pick_order:
                    if pos in partial_comp:
                        brawler = partial_comp[pos]
                        brawler_index = brawler_to_index.get(brawler, unknown_index)
                        indicator = 0 if pos.startswith('a') else 1
                    elif pos == next_pick:
                        brawler_index = brawler_to_index[brawler_name]
                        indicator = 0 if pos.startswith('a') else 1
                    else:
                        brawler_index = unknown_index
                        indicator = 2  # Empty slot
                    sequence.append(brawler_index)
                    indicators.append(indicator)

                input_tensor = torch.LongTensor(sequence).unsqueeze(1).to(device)
                indicator_tensor = torch.LongTensor(indicators).unsqueeze(1).to(device)
                win_rate = model(input_tensor, indicator_tensor).item()
                brawler_win_rates[brawler_name] = win_rate

    return sorted(brawler_win_rates.items(), key=lambda x: x[1], reverse=True)

def cosine_similarity(embeddings):
    # Normalize the vectors
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Calculate cosine similarity
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    return similarity_matrix

def analyze_brawler_embeddings(model, brawler_data):
    embeddings = model.brawler_embedding.weight.detach().cpu().numpy()
    brawler_names = list(brawler_data.keys())

    # Compute cosine similarities
    similarities = cosine_similarity(embeddings)

    # Find most similar brawlers for each brawler
    for i, brawler in enumerate(brawler_names):
        similar_indices = similarities[i].argsort()[::-1][1:6]  # Top 5 most similar, excluding self
        similar_brawlers = [brawler_names[idx] for idx in similar_indices]
        print(f"{brawler}: {', '.join(similar_brawlers)}")

def prepare_data_and_train_model(map_name):
    match_data = prepare_training_data(map=map_name)
    brawler_data = prepare_brawler_data()

    X, team_indicators, y = prepare_sequence_data(match_data, brawler_data)

    debug_brawler_indices(X, brawler_data)

    num_brawlers = len(brawler_data) + 1  # +1 for unknown brawler
    embedding_dim = 128
    nhead = 8
    num_encoder_layers = 4
    dim_feedforward = 256

    model = train_transformer_model(X, team_indicators, y, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
    analyze_brawler_embeddings(model, brawler_data)
    return model, brawler_data
def make_prediction(model, brawler_data, partial_comp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = predict_next_pick(model, partial_comp, brawler_data, device)
    print("Predictions:")
    for brawler, win_rate in predictions[:5]:  # Limiting to top 5 for clarity
        print(f"{brawler}: {win_rate:.4f}")


if __name__ == '__main__':
    # Train the model on the desired map
    model, brawler_data = prepare_data_and_train_model(map_name="Out in the Open")

    # Make a prediction based on a partial composition
    partial_comp = {'a1': 'pearl', 'b1': 'angelo', 'b2': 'piper', 'a2': 'chester'}
    make_prediction(model, brawler_data, partial_comp)
