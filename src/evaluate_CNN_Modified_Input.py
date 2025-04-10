# Force matplotlib to use a non-interactive backend.
import matplotlib
matplotlib.use("Agg")

"""
evaluate_CNN_Modified_Input.py
------------------------------
This script loads the modified CNN model (trained on images that have been cleaned by an aggregated LIME and RF mask),
applies the same mask to the test data, and evaluates the model.
Evaluation metrics and visualizations are saved in a dedicated folder.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Define the CNN model (should match training implementation)
# --------------------------
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
        # For a 224x224 input, after three 2x2 poolings: 224/2/2/2 = 28.
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

# --------------------------
# Load test data and apply mask modification
# --------------------------
def load_test_data(resize_shape=(224, 224)):
    """
    Loads test images and labels from preprocessed .npy files.
    Returns:
      - X_test_norm: float32 images (values in [0,1]) for evaluation.
      - y_test: labels (may be one-hot encoded, which will be converted).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    X_test_norm = X_test.astype(np.float32)  # Assume already in [0,1]
    
    # Convert one-hot labels to class indices if needed.
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    
    return X_test_norm, y_test

# --------------------------
# PyTorch Dataset for Evaluation
# --------------------------
class TorchImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # shape (N, H, W, C)
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

# --------------------------
# Load model checkpoint and meta-data
# --------------------------
def load_model_and_metadata():
    """
    Loads the saved model state dictionary and meta-data JSON from the saved_model folder.
    Returns:
      - state_dict: the saved state dictionary.
      - meta_data: the meta-data dictionary.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    model_path = os.path.join(model_dir, "modified_cnn_model.pth")
    metadata_path = os.path.join(model_dir, "modified_cnn_model_metadata.json")
    
    state_dict = torch.load(model_path, map_location="cpu")
    with open(metadata_path, "r") as f:
        meta_data = json.load(f)
    return state_dict, meta_data

# --------------------------
# Modify images using saved binary mask
# --------------------------
def modify_images(images, binary_mask):
    """
    Applies the binary mask (shape (H, W)) to every image in the input array.
    Assumes images are in [0,1].
    """
    mask_expanded = np.expand_dims(binary_mask, axis=-1)
    modified = images * mask_expanded
    return modified

# --------------------------
# Main Evaluation Function
# --------------------------
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data.
    X_test_norm, y_test = load_test_data((224, 224))
    
    # Load the binary mask (for modified input).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masking_dir = os.path.join(script_dir, "../results/masking_data")
    binary_mask_path = os.path.join(masking_dir, "binary_mask.npy")
    if not os.path.exists(binary_mask_path):
        raise FileNotFoundError("Binary mask not found. Please ensure the training script saved it in ../results/masking_data/")
    binary_mask = np.load(binary_mask_path)
    
    # Modify test images with the saved binary mask.
    modified_X_test = modify_images(X_test_norm, binary_mask)
    
    # Define evaluation transforms.
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Create dataset and loader.
    test_dataset = TorchImageDataset(modified_X_test, y_test, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model and metadata.
    state_dict, meta_data = load_model_and_metadata()
    # Retrieve model parameters from meta-data.
    input_shape = meta_data["model_params"]["input_shape"]  # e.g., [224, 224, 3]
    num_classes = meta_data["model_params"]["num_classes"]
    # Instantiate model (for CNN, use input_channels = input_shape[2]).
    model = CNN(input_shape[2], num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Evaluate model predictions.
    all_preds = []
    all_labels = []
    all_probs = []  # Save softmax probabilities.
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            all_probs.extend(probs.cpu().numpy())
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics.
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    
    # Create results folder for evaluation.
    eval_dir = os.path.join(script_dir, "../results/cnn_modified_eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Save Confusion Matrix Heatmap.
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1", "Class 2"],
                yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Modified CNN Confusion Matrix")
    plt.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
    plt.close("all")
    
    # 2. ROC Curves & AUC (per class).
    y_test_bin = label_binarize(all_labels, classes=[0,1,2])
    n_classes = y_test_bin.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Modified CNN ROC Curves (Per Class)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(eval_dir, "roc_curves.png"))
    plt.close("all")
    
    # 3. Precision-Recall Curves (per class).
    pr_precision = {}
    pr_recall = {}
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        pr_precision[i], pr_recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        plt.plot(pr_recall[i], pr_precision[i], label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Modified CNN Precision-Recall Curves (Per Class)")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(eval_dir, "precision_recall_curves.png"))
    plt.close("all")
    
    # 4. Summary Bar Plot for Overall Metrics.
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [acc, prec, rec, f1]
    plt.figure(figsize=(8,6))
    sns.barplot(x=metric_names, y=metric_values)
    plt.ylim(0, 1)
    plt.title("Modified CNN Summary Metrics")
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.savefig(os.path.join(eval_dir, "summary_metrics.png"))
    plt.close("all")
    
    # 5. Save all evaluation metrics as a JSON file.
    metrics_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": {f"class_{i}": roc_auc[i] for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }
    with open(os.path.join(eval_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Evaluation metrics and plots saved to: {eval_dir}")

if __name__ == "__main__":
    evaluate_model()
