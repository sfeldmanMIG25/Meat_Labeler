"""
Train, tune, and save a VIT transformer model
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import logging

# ---------------------------
# Setup logging and seed
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# Define the Vision Transformer model (simplified)
# ---------------------------
class SimpleViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, depth, heads, mlp_dim):
        super(SimpleViT, self).__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x

# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_data(resize_shape=(224, 224)):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))

    # Convert normalized data to uint8 format
    X_train = (X_train * 255).astype(np.uint8)
    X_test = (X_test * 255).astype(np.uint8)

    X_train_list = [Image.fromarray(img) for img in X_train]
    X_test_list = [Image.fromarray(img) for img in X_test]

    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X_train_tensor = torch.stack([transform(img) for img in X_train_list])
    X_test_tensor = torch.stack([transform(img) for img in X_test_list])

    # Convert one-hot labels into class indices
    y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor

# ---------------------------
# Objective for Optuna hyperparameter tuning
# ---------------------------
def objective(trial):
    # Load preprocessed data
    X_train, y_train, _ = load_data()

    # Split data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Hyperparameter tuning for model architecture and training
    heads = trial.suggest_categorical("heads", [4, 8, 12])
    embed_dim_options = [h * i for h in [4, 8, 12] for i in [32, 64, 128]]
    embed_dim = trial.suggest_categorical("embed_dim", embed_dim_options)
    depth = trial.suggest_int("depth", 4, 12)
    mlp_dim = trial.suggest_categorical("mlp_dim", [512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Ensure embed_dim is divisible by heads
    if embed_dim % heads != 0:
        raise optuna.exceptions.TrialPruned()

    image_size = 224
    patch_size = 16
    num_classes = 3

    # Initialize model
    model = SimpleViT(image_size, patch_size, num_classes, embed_dim, depth, heads, mlp_dim).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 15
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation phase
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        accuracy_val = accuracy_score(targets, preds)
        trial.report(accuracy_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy_val:.4f}")
        best_val_acc = max(best_val_acc, accuracy_val)

    return best_val_acc

# ---------------------------
# Main script: hyperparameter tuning and final training
# ---------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: {}".format(best_trial.value))
    print("  Params:")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # Re-load data and prepare training and validation sets using DataLoader
    X_train, y_train, _ = load_data()
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = best_trial.params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    image_size = 224
    patch_size = 16
    num_classes = 3

    # Filter out only the model-related hyperparameters
    model_param_keys = {'heads', 'embed_dim', 'depth', 'mlp_dim'}
    best_params_model = {k: v for k, v in best_trial.params.items() if k in model_param_keys}

    # Instantiate the best model
    best_model = SimpleViT(image_size, patch_size, num_classes, **best_params_model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 15
    best_val_acc = 0.0
    best_checkpoint = None

    # Final training loop with checkpointing based on validation performance
    for epoch in range(epochs):
        best_model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Final Model Training Epoch {epoch+1}/{epochs}"):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation phase for final model
        best_model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Final Model Validation Epoch {epoch+1}/{epochs}"):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = best_model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        accuracy_val = accuracy_score(targets, preds)
        logging.info(f"Final Model Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy_val:.4f}")
        if accuracy_val > best_val_acc:
            best_val_acc = accuracy_val
            best_checkpoint = best_model.state_dict()

    # Save the best final model checkpoint with meta-data.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_vit_model.pth")
    
    checkpoint = {
        "model_state_dict": best_checkpoint,
        "model_class": "SimpleViT",
        "model_params": {
            "image_size": image_size,
            "patch_size": patch_size,
            "num_classes": num_classes,
            "heads": best_trial.params["heads"],
            "embed_dim": best_trial.params["embed_dim"],
            "depth": best_trial.params["depth"],
            "mlp_dim": best_trial.params["mlp_dim"]
        },
        "best_val_accuracy": best_val_acc,
        "best_trial_params": best_trial.params
    }
    
    torch.save(checkpoint, model_path)
    print(f"Best ViT model saved to {model_path}")
