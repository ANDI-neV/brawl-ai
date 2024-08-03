# we do stuff here
import pandas as pd
import os
import json
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from joblib import dump

def prepare_training_data():
    engine = create_engine('sqlite:///../games.db')

    match_data = pd.read_sql_query("SELECT * FROM battles WHERE map='Out in the Open'", con=engine)

    match_data = match_data.drop(["id", "battleTime", "mode", "map"], axis=1)

    return match_data

def prepare_brawler_data():
    with open('out/brawlers/brawlers.json', 'r') as json_file:
        brawler_data = json.load(json_file)

    return brawler_data
def get_brawler_features(brawler_name, brawler_data, encoder):
    brawler = brawler_data[brawler_name]
    index_encoded = encoder.transform([[brawler['index']]]).toarray()[0]
    print(index_encoded)
    features = {
        'index': index_encoded,
        'movement_speed_normal': brawler['movement speed']['normal'],
        'range_normal': brawler['range']['normal'],
        #'reload_normal': brawler['reload']['normal'], reload feature is only available for 80/82 brawlers
        #'projectile_speed_normal': brawler['projectile speed']['normal'], projectile speed feature is only available for 81/82 brawlers
        f'health': brawler['level stats']['health']['11'],
    }
    return features


def create_brawler_matrix(match_data, brawler_data, limit=None):
    vectors = []
    results = []
    unique_indices = np.array([[brawler['index']] for brawler in brawler_data.values()])
    print(unique_indices)
    encoder = OneHotEncoder()
    encoder.fit(unique_indices)

    if limit is not None:
        match_data = match_data.head(limit)

    for index, row in match_data.iterrows():
        match_vector = []
        match_vector_reversed = []
        for team in ['a', 'b']:
            team_vectors = []
            for i in range(1, 4):
                brawler_col = f'{team}{i}'
                brawler_name = str.lower(row[brawler_col])
                features = get_brawler_features(brawler_name, brawler_data, encoder)
                vector = list(features.values())
                team_vectors.append(vector)
                print(team_vectors)

            team_vectors.sort(key=lambda x: x[0])
            for vector in team_vectors:
                match_vector.extend(vector)

            for vector in reversed(team_vectors):
                match_vector_reversed.extend(vector)

        vectors.append(match_vector)
        results.append(row['result'])
        vectors.append(match_vector_reversed)
        results.append(1 - row['result'])


    vectors = np.array(vectors)
    results = np.array(results)
    return vectors, results



# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        x = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        x = self.dropout4(self.bn4(torch.relu(self.fc4(x))))
        x = self.sigmoid(self.fc5(x))
        return x

class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        confidence_penalty = torch.mean(torch.abs(0.5 - outputs))
        return bce_loss - confidence_penalty

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == targets).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def train_model():
    match_data = prepare_training_data()
    brawler_data = prepare_brawler_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Num GPUs Available: ", torch.cuda.device_count())

    X, y = create_brawler_matrix(match_data, brawler_data)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dump(scaler, 'std_scaler.bin', compress=True)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(X.shape[1])
    print(X)
    model = NeuralNetwork(X.shape[1]).to(device)

    criterion = ConfidenceLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

def prepare_input_data(combination, brawler_data, index_weight=5):
    match_vector = []
    for team in ['a', 'b']:
        team_vectors = []
        for i in range(1, 4):
            brawler_col = f'{team}{i}'
            brawler_name = str.lower(combination[brawler_col])
            features = get_brawler_features(brawler_name, brawler_data)
            features['index'] *= index_weight
            vector = list(features.values())
            team_vectors.append(vector)

        team_vectors.sort(key=lambda x: x[0])
        for vector in team_vectors:
            match_vector.extend(vector)

    match_vector = np.array(match_vector)
    return match_vector.reshape(1, -1)

def normalize_input_data(input_data, scaler):
    normalized_data = scaler.transform(input_data)
    return normalized_data

def predict_win_probability(model, input_data, device):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        output = model(input_tensor)
        probability = output.cpu().numpy()[0][0]
    return probability

train_model()