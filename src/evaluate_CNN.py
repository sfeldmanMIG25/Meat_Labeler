"""
Use metadata to recreate and evaluate the basic CNN model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class
class MeatFreshnessDataset(Dataset):
    def __init__(self, X, y):
        # Convert images from H x W x C to tensor of shape [N, C, H, W]
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load preprocessed data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    # If y_test is one-hot encoded, convert to class indices.
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    return X_train, y_train, X_test, y_test

# Define the CNN model (must match training)
class CNN(nn.Module):
    def __init__(self, conv_filters1, conv_filters2, dropout_rate):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, conv_filters1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv_filters1, conv_filters2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv_filters2, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.layers(x)

# Mapping for future extension if additional models are used
model_classes = {
    "CNN": CNN,
}

# Generalized evaluation function with extra metrics and plots
def evaluate_model(model_path="meat_freshness_cnn_tuned.pt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    full_model_path = os.path.join(model_dir, model_path)
    
    # Load the saved checkpoint
    checkpoint = torch.load(full_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_class_name = checkpoint.get("model_class", "CNN")
        model_params = checkpoint.get("model_params", {})
        print(f"Loaded checkpoint with meta-data:\n  Model class: {model_class_name}\n  Model params: {model_params}")
    else:
        # Fallback if meta-data is not available
        state_dict = checkpoint
        print("Warning: Checkpoint does not contain meta-data. Falling back to default CNN parameters.")
        conv_filters1 = state_dict["layers.0.weight"].shape[0] if "layers.0.weight" in state_dict else 32
        conv_filters2 = state_dict["layers.3.weight"].shape[0] if "layers.3.weight" in state_dict else 64
        dropout_rate = 0.3
        model_class_name = "CNN"
        model_params = {"conv_filters1": conv_filters1, "conv_filters2": conv_filters2, "dropout_rate": dropout_rate}
    
    if model_class_name in model_classes:
        ModelClass = model_classes[model_class_name]
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
    
    model = ModelClass(**model_params).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test data
    _, _, X_test, y_test = load_data()
    test_dataset = MeatFreshnessDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []  # To store probability outputs
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = softmax(outputs)  # Get probabilities for each class
            all_probs.extend(probabilities.cpu().numpy())
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Compute standard metrics
    acc = accuracy_score(all_targets, all_preds)
    cl_report = classification_report(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    # Create results directory
    results_dir = os.path.join(script_dir, "../results/evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Save classification report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(cl_report)

    # Save model info and standard metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_summary = (
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Total Parameters: {total_params}\n"
        f"Trainable Parameters: {trainable_params}\n"
        f"Model Architecture:\n{model}\n"
    )
    with open(os.path.join(results_dir, "model_info.txt"), "w") as f:
        f.write(metrics_summary)

    # -----------------------------
    # ROC and AUC Calculations
    # -----------------------------
    # Binarize the true labels for ROC analysis (assuming 3 classes)
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(all_targets, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (per Class)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curves.png"))
    plt.close()

    # -----------------------------
    # Precision-Recall Curves
    # -----------------------------
    pr_precision = dict()
    pr_recall = dict()
    for i in range(n_classes):
        pr_precision[i], pr_recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(pr_recall[i], pr_precision[i], label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (per Class)")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, "precision_recall_curves.png"))
    plt.close()

    # -----------------------------
    # Summary Bar Plot for Metrics
    # -----------------------------
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_values = [acc, precision, recall, f1]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics_names, y=metrics_values)
    plt.ylim(0, 1)
    plt.title("Summary Metrics")
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(results_dir, "summary_metrics.png"))
    plt.close()

    # Save all computed metrics to a JSON file
    import json
    metrics_dict = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": {f"class_{i}": roc_auc[i] for i in range(n_classes)},
        "confusion_matrix": conf_matrix.tolist()
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print("Evaluation complete. Results saved to:", results_dir)

if __name__ == "__main__":
    evaluate_model()
