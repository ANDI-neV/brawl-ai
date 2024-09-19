import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
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

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        return x + self.positional_embedding(positions)


class BrawlStarsTransformer1(nn.Module):
    def __init__(self, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ally_brawler_embedding = nn.Embedding(num_brawlers, embedding_dim)
        self.enemy_brawler_embedding = nn.Embedding(num_brawlers, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = LearnedPositionalEncoding(embedding_dim, max_len=6)
        self.attention = nn.MultiheadAttention(embedding_dim, nhead, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, src, team_indicators):
        ally_mask = (team_indicators == 0)
        enemy_mask = (team_indicators == 1)

        ally_embeddings = self.dropout(self.ally_brawler_embedding(src)) * ally_mask.unsqueeze(-1)
        enemy_embeddings = self.dropout(self.enemy_brawler_embedding(src)) * enemy_mask.unsqueeze(-1)
        src = ally_embeddings + enemy_embeddings
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        attn_output, attn_weights = self.attention(src, src, src)
        print(attn_weights)
        attn_output = attn_output.transpose(0, 1)
        output = self.transformer_encoder(attn_output)
        output = self.dropout(output)
        return torch.sigmoid(self.decoder(output.mean(dim=0)))

class BrawlStarsTransformer(nn.Module):
    def __init__(self, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ally_brawler_embedding = nn.Embedding(num_brawlers, embedding_dim)
        self.enemy_brawler_embedding = nn.Embedding(num_brawlers, embedding_dim)
        self.pos_encoder = LearnedPositionalEncoding(embedding_dim, max_len=6)
        self.attention = nn.MultiheadAttention(embedding_dim, nhead, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, src, team_indicators):
        ally_mask = (team_indicators == 0)
        enemy_mask = (team_indicators == 1)

        ally_embeddings = self.ally_brawler_embedding(src) * ally_mask.unsqueeze(-1)
        enemy_embeddings = self.enemy_brawler_embedding(src) * enemy_mask.unsqueeze(-1)
        src = ally_embeddings + enemy_embeddings

        # Positional Encoding
        src = self.pos_encoder(src)

        # Self-Attention Mechanism
        src = src.transpose(0, 1)
        attn_output, _ = self.attention(src, src, src)
        attn_output = self.layer_norm(attn_output + src)  # Add and Norm
        attn_output = attn_output.transpose(0, 1)

        # Transformer Encoder
        output = self.transformer_encoder(attn_output)
        output = self.dropout(output)

        # Output
        return torch.sigmoid(self.decoder(output.mean(dim=0)))


class BrawlStarsDataset(Dataset):
    def __init__(self, X, team_indicators, y):
        self.X = torch.LongTensor(X)
        self.team_indicators = torch.LongTensor(team_indicators)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.team_indicators[idx], self.y[idx]



def validate_partial_comp(partial_comp):
    valid_order = ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']
    for i, key in enumerate(partial_comp.keys()):
        if key != valid_order[i]:
            return False
    return True

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)


def prepare_sequence_data(match_data, brawler_data):
    sequences = []
    team_indicators = []
    results = []

    brawler_to_index = {name: idx for idx, name in enumerate(brawler_data.keys())}
    unknown_index = len(brawler_to_index)

    for _, row in match_data.iterrows():
        sequence = []
        indicators = []
        for pos in ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']:
            brawler_name = str.lower(row[pos])
            brawler_index = brawler_to_index.get(brawler_name, unknown_index)
            sequence.append(brawler_index)
            indicators.append(0 if pos.startswith('a') else 1)

        sequences.append(sequence)
        team_indicators.append(indicators)
        results.append(float(row['result']))

    return np.array(sequences), np.array(team_indicators), np.array(results)


def train_transformer_model(X, team_indicators, y, num_brawlers, embedding_dim, nhead, num_encoder_layers,
                            dim_feedforward,
                            batch_size=64, epochs=50, learning_rate=0.001):
    X_train, X_val, team_indicators_train, team_indicators_val, y_train, y_val = train_test_split(
        X, team_indicators, y, test_size=0.2, random_state=42)

    train_dataset = BrawlStarsDataset(X_train, team_indicators_train, y_train)
    val_dataset = BrawlStarsDataset(X_val, team_indicators_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BrawlStarsTransformer(num_brawlers + 1, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_team_indicators, batch_y in train_loader:
            batch_X, batch_team_indicators, batch_y = batch_X.to(device), batch_team_indicators.to(device), batch_y.to(
                device)

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
                batch_X, batch_team_indicators, batch_y = batch_X.to(device), batch_team_indicators.to(
                    device), batch_y.to(device)
                outputs = model(batch_X.transpose(0, 1), batch_team_indicators.transpose(0, 1))
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model


def predict_winrate(model, partial_comp, brawler_data, device):
    model.eval()
    with torch.no_grad():
        brawler_to_index = {name: idx for idx, name in enumerate(brawler_data.keys())}
        unknown_index = len(brawler_to_index)

        sequence = []
        indicators = []
        for pos in ['a1', 'b1', 'b2', 'a2', 'a3', 'b3']:
            if pos in partial_comp:
                brawler = partial_comp[pos]
                brawler_index = brawler_to_index.get(brawler, unknown_index)
                indicator = 0 if pos.startswith('a') else 1
            else:
                brawler_index = unknown_index
                indicator = 1 if pos.startswith('b') else 0  # Assume unknown enemy for 'b', ally for 'a'
            sequence.append(brawler_index)
            indicators.append(indicator)

        input_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
        indicator_tensor = torch.LongTensor(indicators).unsqueeze(0).to(device)

        win_probability = model(input_tensor, indicator_tensor).item()

    return win_probability

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

    num_brawlers = len(brawler_data)
    embedding_dim = 128
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 256

    model = train_transformer_model(X, team_indicators, y, num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
    analyze_brawler_embeddings(model, brawler_data)
    return model, brawler_data

def get_model():
    brawler_data = prepare_brawler_data()

    num_brawlers = len(brawler_data) + 1
    embedding_dim = 128
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 256

    model = BrawlStarsTransformer(num_brawlers, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
    model.apply(initialize_weights)

    # Load the state dict
    state_dict = torch.load('best_transformer_model.pth')

    # Check if the state dict matches the model architecture
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    return model


def make_prediction(partial_comp, map_name):
    model = get_model()
    brawler_data = prepare_brawler_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    win_probability = predict_winrate(model, partial_comp, brawler_data, device)
    print(f"Win Probability: {win_probability:.4f}")

    best_picks = []
    for brawler in brawler_data.keys():
        if brawler not in partial_comp.values():
            next_pick = next(pos for pos in ['a1', 'b1', 'b2', 'a2', 'a3', 'b3'] if pos not in partial_comp)
            new_comp = partial_comp.copy()
            new_comp[next_pick] = brawler
            prob = predict_winrate(model, new_comp, brawler_data, device)
            best_picks.append((brawler, prob))

    best_picks.sort(key=lambda x: x[1], reverse=True)
    print("\nBest next picks:")
    for brawler, prob in best_picks[:5]:
        print(f"{brawler}: {prob:.4f}")

    return win_probability, best_picks


if __name__ == '__main__':
    prepare_data_and_train_model("Out in the Open")
    partial_comp = {'a1': 'angelo', 'b1': 'pearl', 'b2': 'chester', 'a2': 'piper'}
    make_prediction(partial_comp, "Out in the Open")