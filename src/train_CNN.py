"""
Train, tune, save the simple CNN model for the image data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import random
import optuna
from tqdm import tqdm

# Set seeds for reproducibility
def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom Dataset class
class MeatFreshnessDataset(Dataset):
    def __init__(self, X, y):
        # Convert images to float tensors and rearrange dimensions: [N, H, W, C] -> [N, C, H, W]
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        # Convert one-hot labels to class indices
        self.y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load preprocessed data from .npy files
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    return X_train, y_train, X_test, y_test

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, conv_filters1, conv_filters2, dropout_rate):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv_filters1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=conv_filters1, out_channels=conv_filters2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=conv_filters2, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=128 * 26 * 26, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=3)
        )

    def forward(self, x):
        return self.layers(x)

# Objective function for Optuna hyperparameter tuning
def objective(trial):
    X_train, y_train, _, _ = load_data()
    dataset = MeatFreshnessDataset(X_train, y_train)

    # Create training/validation split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    conv_filters1 = trial.suggest_int("conv_filters1", 32, 128, step=32)
    conv_filters2 = trial.suggest_int("conv_filters2", 64, 256, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    model = CNN(conv_filters1, conv_filters2, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy

# Tune hyperparameters and train the final model
def tune_and_train_model():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # Adjust n_trials as needed

    best_params = study.best_params
    print("Best parameters:", best_params)

    # Reload the full dataset
    X_train, y_train, X_test, y_test = load_data()
    train_dataset = MeatFreshnessDataset(X_train, y_train)
    test_dataset = MeatFreshnessDataset(X_test, y_test)

    # Create training/validation split from the training data
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

    model = CNN(best_params["conv_filters1"], best_params["conv_filters2"], best_params["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params["optimizer"])(model.parameters(), lr=best_params["learning_rate"])

    epochs = 20  # Train for longer with the best parameters
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validate after each epoch
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_model_state = model.state_dict()

    # Save checkpoint with metadata for later use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_model_state,
        "model_class": "CNN",
        "model_params": {
            "conv_filters1": best_params["conv_filters1"],
            "conv_filters2": best_params["conv_filters2"],
            "dropout_rate": best_params["dropout_rate"],
        },
        "best_val_accuracy": best_val_acc,
        "best_params": best_params,
    }
    model_path = os.path.join(model_dir, "meat_freshness_cnn_tuned.pt")
    torch.save(checkpoint, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    set_random_seed()
    tune_and_train_model()
