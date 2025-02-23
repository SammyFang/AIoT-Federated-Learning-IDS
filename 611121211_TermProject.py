import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Local training process
def local_train(model, train_loader, criterion, optimizer, epochs, mu=0, global_model=None):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if mu > 0 and global_model is not None:
                proximal_term = 0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

# Federated averaging algorithm (FedAvg)
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

# Federated proximal algorithm (FedProx)
def federated_proximal(global_model, client_models, mu):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

# Meta sampling algorithm
class MetaSampler(nn.Module):
    def __init__(self, input_size):
        super(MetaSampler, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

    def sample(self, data, weights):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=torch.float32)
        probabilities = self.forward(data)
        selected_indices = torch.multinomial(probabilities.flatten(), num_samples=int(weights.sum().item()))
        return data[selected_indices], weights[selected_indices]

def train_meta_sampler(meta_sampler, data, weights, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(meta_sampler.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        sampled_data, sampled_weights = meta_sampler.sample(data, weights)
        output = meta_sampler(sampled_data)
        loss = criterion(output, sampled_weights)
        loss.backward()
        optimizer.step()

    return meta_sampler

# Client class
class Client:
    def __init__(self, data, target, batch_size, learning_rate, epochs):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        tensor_data = torch.tensor(data, dtype=torch.float32)
        tensor_target = torch.tensor(target, dtype=torch.long)
        self.train_loader = DataLoader(TensorDataset(tensor_data, tensor_target), batch_size=batch_size, shuffle=True)

        self.model = MLP(input_size=data.shape[1], hidden_size=128, output_size=2)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model=None, mu=0):
        local_train(self.model, self.train_loader, self.criterion, self.optimizer, self.epochs, mu, global_model)
        return self.model

# Dynamic clustering algorithm
def dynamic_clustering(models, num_clusters=5):
    model_params = [model.state_dict() for model in models]
    model_vectors = [torch.cat([param.flatten() for param in model.values()]) for model in model_params]
    model_matrix = torch.stack(model_vectors).numpy()

    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(model_matrix)
    labels = kmeans.labels_

    clustered_models = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_models = [models[j] for j in cluster_indices]
        aggregated_model = aggregate_models(cluster_models)
        clustered_models.append(aggregated_model)

    return clustered_models

def aggregate_models(models):
    global_dict = models[0].state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
    aggregated_model = MLP(input_size=global_dict['fc1.weight'].shape[1], hidden_size=128, output_size=2)
    aggregated_model.load_state_dict(global_dict)
    return aggregated_model

# Plot training and validation curves
def plot_training_curves(train_losses, val_f1_scores, val_auc_scores):
    epochs = range(len(train_losses))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_scores, 'b', label='Validation F1 Score')
    plt.plot(epochs, val_auc_scores, 'g', label='Validation AUC Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.show()

# Local training process
def local_train(model, train_loader, criterion, optimizer, epochs, mu=0, global_model=None):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if mu > 0 and global_model is not None:
                proximal_term = 0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")

# Main function
def main():
    print("Loading dataset...")
    # Load dataset
    file_path = '//NSL_KDD_Train.csv'
    data = pd.read_csv(file_path)
    print("Dataset loaded.")

    # Define columns for preprocessing
    categorical_features = [1, 2, 3]  # assuming the 1st, 2nd, 3rd columns are categorical
    numeric_features = list(set(range(data.shape[1] - 1)) - set(categorical_features))  # Exclude the label column

    # Separate features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].apply(lambda x: 1 if x == 'normal' else 0)

    # Preprocess pipeline for categorical and numeric data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    # Apply preprocessing
    X = preprocessor.fit_transform(X)
    print("Preprocessing done.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train-test split done.")

    # Ensure labels are NumPy arrays
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

    clients = [Client(X_train, y_train, batch_size=32, learning_rate=0.01, epochs=5) for _ in range(10)]
    global_model = MLP(input_size=X_train.shape[1], hidden_size=128, output_size=2)
    meta_sampler = MetaSampler(input_size=X_train.shape[1])
    print("Clients and models initialized.")

    train_losses = []
    val_f1_scores = []
    val_auc_scores = []

    for round in range(5):
        print(f"Starting round {round + 1}...")
        client_models = []
        for client in clients:
            print(f"Training client {clients.index(client) + 1}...")
            sampled_data, sampled_weights = meta_sampler.sample(client.data, torch.ones(client.data.shape[0]))
            client.data = sampled_data.numpy()
            client.target = sampled_weights.numpy()
            client_models.append(client.train(global_model=global_model, mu=0.1))  # Using FedProx algorithm

        clustered_models = dynamic_clustering(client_models)
        federated_proximal(global_model, clustered_models, mu=0.1)

        global_model.eval()
        val_tensor = torch.tensor(X_test, dtype=torch.float32)
        val_target_tensor = torch.tensor(y_test, dtype=torch.long)
        with torch.no_grad():
            val_outputs = global_model(val_tensor).argmax(dim=1).numpy()
        val_f1 = f1_score(val_target_tensor.numpy(), val_outputs)
        val_auc = roc_auc_score(val_target_tensor.numpy(), val_outputs)
        val_f1_scores.append(val_f1)
        val_auc_scores.append(val_auc)

        # Calculate training loss
        train_loss = np.mean([client.criterion(client.model(torch.tensor(client.data, dtype=torch.float32)), torch.tensor(client.target, dtype=torch.long)).item() for client in clients])
        train_losses.append(train_loss)

        print(f'Round {round + 1}: F1 Score: {val_f1}, AUC: {val_auc}')

    plot_training_curves(train_losses, val_f1_scores, val_auc_scores)
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
