"""
Evaluate the best VIT model generated in train_VIT.py. Print and save results for future reference.
"""


import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Define the Vision Transformer model (SimpleViT)
# --------------------------
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
        x = self.patch_embed(x)            # shape: [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)     # shape: [B, num_patches, embed_dim]
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches+1, embed_dim]
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x

# --------------------------
# Data Loading & Preprocessing for Evaluation
# --------------------------
def load_test_data(resize_shape=(224, 224)):
    """
    Loads test images and labels from preprocessed .npy files.
    The images (which are normalized floats during training and then multiplied by 255) 
    are converted to uint8 and then to PIL images. A deterministic transform (Resize, ToTensor, Normalize)
    is applied before evaluation. One-hot labels (if any) are converted to class indices.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Convert images back to uint8.
    X_test = (X_test * 255).astype(np.uint8)
    X_test_list = [Image.fromarray(img) for img in X_test]
    
    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    X_test_tensor = torch.stack([transform(img) for img in X_test_list])
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    
    return X_test_tensor, y_test

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------
# Load ViT Checkpoint with Meta-data
# --------------------------
def load_vit_checkpoint(model_filename="best_vit_model.pth"):
    """
    Loads the stored ViT checkpoint (with meta-data) from the saved_model folder.
    Expected keys: "model_state_dict", "model_class", "model_params".
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    full_model_path = os.path.join(model_dir, model_filename)
    
    checkpoint = torch.load(full_model_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_class = checkpoint.get("model_class", "SimpleViT")
        model_params = checkpoint.get("model_params", {})
        print(f"Loaded ViT checkpoint with meta-data:\n  Model class: {model_class}\n  Model parameters: {model_params}")
    else:
        state_dict = checkpoint
        print("Warning: Checkpoint missing meta-data. Falling back to default parameters.")
        model_class = "SimpleViT"
        # These default parameters must match what was used during training.
        model_params = {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 3,
            "embed_dim": 768,
            "depth": 8,
            "heads": 12,
            "mlp_dim": 2048
        }
    return state_dict, model_class, model_params

# --------------------------
# Main Evaluation Function for ViT
# --------------------------
def evaluate_vit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data and create dataset and DataLoader.
    X_test_tensor, y_test = load_test_data((224, 224))
    dataset = ViTDataset(X_test_tensor, y_test)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load the checkpoint and instantiate the model.
    state_dict, model_class, model_params = load_vit_checkpoint()
    if model_class == "SimpleViT":
        model = SimpleViT(**model_params).to(device)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # For storing softmax probabilities.
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
    
    # Compute evaluation metrics.
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
    
    # Create a directory for evaluation results.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results/vit_eval")
    os.makedirs(results_dir, exist_ok=True)
    
    # ------------------------------
    # 1. Confusion Matrix Heatmap.
    # ------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fresh", "Half-Fresh", "Spoiled"],
                yticklabels=["Fresh", "Half-Fresh", "Spoiled"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ViT Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # ------------------------------
    # 2. ROC Curves & AUC per Class.
    # ------------------------------
    # Binarize the true labels for ROC analysis.
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ViT ROC Curves (Per Class)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curves.png"))
    plt.close()
    
    # ------------------------------
    # 3. Precision-Recall Curves (Per Class).
    # ------------------------------
    pr_precision = {}
    pr_recall = {}
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        pr_precision[i], pr_recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        plt.plot(pr_recall[i], pr_precision[i], label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ViT Precision-Recall Curves (Per Class)")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, "precision_recall_curves.png"))
    plt.close()
    
    # ------------------------------
    # 4. Summary Bar Plot for Overall Metrics.
    # ------------------------------
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [acc, prec, rec, f1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metric_names, y=metric_values)
    plt.ylim(0, 1)
    plt.title("ViT Summary Metrics")
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.savefig(os.path.join(results_dir, "summary_metrics.png"))
    plt.close()
    
    # ------------------------------
    # 5. Save All Evaluation Metrics to JSON.
    # ------------------------------
    metrics_dict = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": {f"class_{i}": roc_auc[i] for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"ViT evaluation metrics and plots are saved to: {results_dir}")

if __name__ == "__main__":
    evaluate_vit()
