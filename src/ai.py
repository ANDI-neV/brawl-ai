import json
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from joblib import dump

def prepare_training_data():
    engine = create_engine('sqlite:///../games.db')

    match_data = pd.read_sql_query("SELECT * FROM battles WHERE map='Undermine'", con=engine)
    match_data = match_data.drop(["id", "battleTime", "mode", "map"], axis=1)

    return match_data

def prepare_brawler_data():
    with open('out/brawlers/brawlers.json', 'r') as json_file:
        brawler_data = json.load(json_file)

    return brawler_data

class BrawlStarsNN(nn.Module):
    def __init__(self, input_size):
        super(BrawlStarsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def create_brawler_matrix(match_data, brawler_data, scaler, encoder, limit=None):
    vectors = []
    results = []

    if limit is not None:
        match_data = match_data.head(limit)

    for index, row in match_data.iterrows():
        match_vector = []
        for team in ['a', 'b']:
            team_vector = []
            for i in range(1, 4):
                brawler_col = f'{team}{i}'
                brawler_name = str.lower(row[brawler_col])
                brawler = brawler_data[brawler_name]

                # One-hot encode brawler index
                index_encoded = encoder.transform([[brawler['index']]]).toarray()[0]

                # Get continuous features
                continuous_features = [
                    brawler['movement speed']['normal'],
                    brawler['range']['normal'],
                    brawler['level stats']['health']['11']
                ]
                continuous_features_scaled = scaler.transform([continuous_features])[0]

                # Combine features
                brawler_vector = np.concatenate((index_encoded, continuous_features_scaled))
                team_vector.extend(brawler_vector)

            match_vector.extend(team_vector)

        vectors.append(match_vector)
        results.append(row['result'])

    return np.array(vectors), np.array(results)


def train_model(X, y, epochs=80, batch_size=64, learning_rate=0.001):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BrawlStarsNN(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_accuracy = ((train_outputs > 0.5) == y_train_tensor).float().mean()

            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_accuracy = ((test_outputs > 0.5) == y_test_tensor).float().mean()

        scheduler.step(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print("---")

    return model

if __name__ == '__main__':
    match_data = prepare_training_data()
    brawler_data = prepare_brawler_data()

    # Prepare encoders and scalers
    unique_indices = np.array([[brawler['index']] for brawler in brawler_data.values()])
    encoder = OneHotEncoder()
    encoder.fit(unique_indices)

    continuous_features = [
        [brawler['movement speed']['normal'], brawler['range']['normal'], brawler['level stats']['health']['11']]
        for brawler in brawler_data.values()
    ]
    scaler = StandardScaler()
    scaler.fit(continuous_features)

    dump(encoder, 'encoder.joblib')
    dump(scaler, 'scaler.joblib')

    # Create feature matrix
    X, y = create_brawler_matrix(match_data, brawler_data, scaler, encoder)

    # Train the model
    model = train_model(X, y)

    # Save the model
    torch.save(model.state_dict(), 'brawl_stars_model_undermine.pth')


def get_brawler_features(brawler_name, brawler_data):
    brawler = brawler_data[brawler_name]
    continuous_features = [
        brawler['movement speed']['normal'],
        brawler['range']['normal'],
        brawler['level stats']['health']['11']
    ]
    return brawler['index'], continuous_features


def prepare_input_data(combination, brawler_data, encoder, scaler):
    match_vector = []
    for team in ['a', 'b']:
        for i in range(1, 4):
            brawler_col = f'{team}{i}'
            brawler_name = str.lower(combination[brawler_col])
            brawler_index, continuous_features = get_brawler_features(brawler_name, brawler_data)

            # One-hot encode brawler index
            index_encoded = encoder.transform([[brawler_index]]).toarray()[0]

            # Scale continuous features
            continuous_features_scaled = scaler.transform([continuous_features])[0]

            # Combine features
            brawler_vector = np.concatenate((index_encoded, continuous_features_scaled))
            match_vector.extend(brawler_vector)

    return np.array(match_vector).reshape(1, -1)


def predict_win_probability(model, input_data, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).to(device)
        output = model(input_tensor)
        probability = output.item()
    return probability