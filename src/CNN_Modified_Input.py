# Force Matplotlib to use a non-interactive backend to avoid Tkinter issues.
import matplotlib
matplotlib.use("Agg")

"""
CNN_Modified_Input.py
----------------------
This script uses interpretability methods (LIME and RandomForest) to generate an aggregated importance mask.
That mask is applied to input images to remove unimportant noise.
The modified images are then used to train a CNN model in PyTorch.
The training uses progress bars and the best model (with meta-data) is saved using torch.save().
Additionally, the combined masking data is saved to facilitate easy future evaluation.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import json

# ---------------------------
# Set random seeds for repeatability
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_data(resize_shape=(224, 224)):
    """
    Loads preprocessed data from .npy files.
    Returns:
      - X_train_norm: float32 images in [0,1] for training.
      - y_train: one-hot encoded labels.
      - X_test_norm: float32 images in [0,1] for evaluation.
      - X_train_uint8, X_test_uint8: uint8 versions (for LIME/RF explanations).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    
    X_train_norm = X_train.astype(np.float32)  # assuming already normalized in [0,1]
    X_test_norm = X_test.astype(np.float32)
    
    # Create uint8 copies for LIME/RF (scale from [0,1] to [0,255])
    X_train_uint8 = (X_train_norm * 255).astype(np.uint8)
    X_test_uint8 = (X_test_norm * 255).astype(np.uint8)
    
    return X_train_norm, y_train, X_test_norm, X_train_uint8, X_test_uint8

# ---------------------------
# Define PyTorch CNN Model
# ---------------------------
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # For 224x224 input, after three 2x2 poolings: 224/2/2/2 = 28.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# LIME Explanation Functions
# ---------------------------
def predict_fn(images, model, device):
    """
    Prediction function for LIME.
    Expects images in shape (N, H, W, C) in uint8; applies a deterministic transform.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    model.eval()
    batch = []
    for img in images:
        img_t = transform(img)
        batch.append(img_t)
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

def explain_cnn(model, image, index, save_dir, device):
    """
    Uses LIME to generate an explanation for a single image.
    Saves a visualization and the raw mask, then returns the mask.
    The input image should be a uint8 numpy array with shape (H, W, C).
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        lambda imgs: predict_fn(imgs, model, device),
        top_labels=1,
        hide_color=0,
        num_samples=100
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=5,
        hide_rest=False
    )
    temp_norm = np.clip(temp/255.0, 0, 1)
    plt.imshow(mark_boundaries(temp_norm, mask))
    plt.savefig(os.path.join(save_dir, f"lime_explanation_{index}.png"))
    plt.close("all")
    np.save(os.path.join(save_dir, f"lime_mask_{index}.npy"), mask)
    return mask

# ---------------------------
# RandomForest Importance Explanation
# ---------------------------
def explain_rf(X_train_flat, y_train, image_shape, save_dir):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train
    rf.fit(X_train_flat, y_train_labels)
    importances = rf.feature_importances_
    # Reshape to full image shape (H, W, 3) then average over channels
    full_shape = (image_shape[0], image_shape[1], 3)
    importances = importances.reshape(full_shape).mean(axis=-1)
    np.save(os.path.join(save_dir, "rf_importance.npy"), importances)
    return importances

# ---------------------------
# Combine LIME and RF Masks into a Single Mask
# ---------------------------
def combine_masks(lime_masks, rf_mask, weight=0.5):
    aggregated_lime = np.mean(np.array(lime_masks), axis=0)
    aggregated_lime_norm = (aggregated_lime - aggregated_lime.min()) / (aggregated_lime.ptp() + 1e-8)
    rf_mask_norm = (rf_mask - rf_mask.min()) / (rf_mask.ptp() + 1e-8)
    combined = weight * aggregated_lime_norm + (1 - weight) * rf_mask_norm
    binary_mask = (combined >= 0.5).astype(np.float32)
    return combined, binary_mask

# ---------------------------
# Modify Images using the Binary Mask
# ---------------------------
def modify_images(images, binary_mask):
    mask_expanded = np.expand_dims(binary_mask, axis=-1)
    modified = images * mask_expanded
    return modified

# ---------------------------
# PyTorch Dataset for Training/Evaluation
# ---------------------------
class TorchImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # shape (N, H, W, C) in [0,1]
        self.labels = labels  # class indices
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            image = torch.tensor(image).permute(2,0,1).float()
        label = self.labels[idx]
        return image, label

# ---------------------------
# Training and Evaluation with Progress Bars
# ---------------------------
def train_and_evaluate_cnn(train_images, train_labels, val_images, val_labels, input_shape, num_classes, num_epochs=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_dataset = TorchImageDataset(train_images, train_labels, transform=transform)
    val_dataset = TorchImageDataset(val_images, val_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = CNN(input_shape[2], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_dataset)
        
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation")
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    return model, best_state

# ---------------------------
# Main Workflow
# ---------------------------
if __name__ == "__main__":
    # Load data.
    X_train_norm, y_train, X_test_norm, X_train_uint8, X_test_uint8 = load_data()
    
    # Compute consistent train/validation indices.
    num_samples = X_train_norm.shape[0]
    indices = np.arange(num_samples)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train_norm_split = X_train_norm[train_idx]
    y_train_split = y_train[train_idx]
    X_val_norm = X_train_norm[val_idx]
    y_val = y_train[val_idx]
    
    # Also split the uint8 version accordingly.
    X_train_uint8_split = X_train_uint8[train_idx]
    
    input_shape = X_train_norm_split.shape[1:]   # e.g., (224,224,3)
    num_classes = y_train_split.shape[1]           # assuming one-hot encoded labels
    
    # Create directories for saving explanation results and masking data.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results")
    lime_dir = os.path.join(results_dir, "lime_results")
    os.makedirs(lime_dir, exist_ok=True)
    rf_dir = os.path.join(results_dir, "rf_results")
    os.makedirs(rf_dir, exist_ok=True)
    masking_dir = os.path.join(results_dir, "masking_data")
    os.makedirs(masking_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------
    # 1. Generate LIME masks on a sample of training images.
    baseline_model = CNN(input_shape[2], num_classes).to(device)  # Untrained model for explanation.
    num_sample = 10
    lime_masks = []
    for idx in range(num_sample):
        image = X_train_uint8_split[idx]  # Shape (224,224,3) uint8.
        mask = explain_cnn(baseline_model, image, idx, lime_dir, device)
        lime_masks.append(mask)
    
    # ---------------------------
    # 2. Compute RF importance mask using the uint8 training set.
    image_shape = X_train_uint8_split[0].shape[:2]  # (224,224)
    X_train_flat = X_train_uint8_split.reshape(X_train_uint8_split.shape[0], -1)
    rf_mask = explain_rf(X_train_flat, y_train_split, image_shape, rf_dir)
    
    # ---------------------------
    # 3. Combine LIME masks and RF importance mask.
    combined_mask, binary_mask = combine_masks(lime_masks, rf_mask, weight=0.5)
    np.save(os.path.join(masking_dir, "combined_mask.npy"), combined_mask)
    np.save(os.path.join(masking_dir, "binary_mask.npy"), binary_mask)
    
    # ---------------------------
    # 4. Modify images using the binary mask.
    modified_X_train = modify_images(X_train_norm_split, binary_mask)
    modified_X_val = modify_images(X_val_norm, binary_mask)
    modified_X_test = modify_images(X_test_norm, binary_mask)
    
    # ---------------------------
    # 5. Train and evaluate the new CNN on modified images.
    y_train_labels = np.argmax(y_train_split, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    trained_model, best_state = train_and_evaluate_cnn(modified_X_train, y_train_labels, modified_X_val, y_val_labels, input_shape, num_classes, num_epochs=20, batch_size=32)
    
    # ---------------------------
    # 6. Save the new CNN model and meta-data (including mask paths) using torch.save.
    model_dir = os.path.join(results_dir, "saved_model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "modified_cnn_model.pth")
    
    meta_data = {
        "model_class": "CNN",
        "model_params": {
            "input_shape": input_shape,
            "num_classes": num_classes
        },
        "training_hyperparameters": {
            "epochs": 20,
            "optimizer": "Adam",
            "batch_size": 32
        },
        "masking_data": {
            "combined_mask_path": os.path.join("..", "masking_data", "combined_mask.npy"),
            "binary_mask_path": os.path.join("..", "masking_data", "binary_mask.npy")
        },
        "notes": "Images modified using aggregated LIME and RF masks to remove noise."
    }
    
    torch.save(best_state, model_path)
    with open(os.path.join(model_dir, "modified_cnn_model_metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=4)
    
    print(f"Modified CNN model saved to {model_path}")
    print("Meta-data saved as modified_cnn_model_metadata.json")
