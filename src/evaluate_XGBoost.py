"""
Import, test and store evaluation results of the best XGBoost classifier from train_XGBoost.py
"""

import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2

def load_data(resize_shape=(32, 32)):
    """
    Loads X_test and y_test from preprocessed .npy files.
    Resizes images and flattens them for prediction.
    Converts one-hot labels to class indices if needed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Resize each image and flatten
    X_test_resized = np.array([cv2.resize(img, resize_shape) for img in X_test])
    X_test_flat = X_test_resized.reshape(X_test_resized.shape[0], -1)
    
    # Convert one-hot labels to indices if necessary
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    
    return X_test_flat, y_test

def load_model_and_params():
    """
    Loads the saved XGBoost model and meta-data (best parameters).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    params_dir = os.path.join(script_dir, "../results")
    
    model_path = os.path.join(model_dir, "xgboost_model.json")
    params_path = os.path.join(params_dir, "xgboost_best_params.json")
    
    booster = xgb.Booster()
    booster.load_model(model_path)
    
    with open(params_path, "r") as f:
        meta_data = json.load(f)
    
    return booster, meta_data

def evaluate_xgboost():
    # Load test data and the saved model/meta-data.
    X_test, y_test = load_data()
    booster, meta = load_model_and_params()
    
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Use iteration_range (instead of ntree_limit) to predict using best_iteration.
    best_iter = meta.get("best_iteration", 0)
    preds_prob = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    preds = np.argmax(preds_prob, axis=1)
    
    # Calculate standard metrics.
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    conf_matrix = confusion_matrix(y_test, preds)
    class_report = classification_report(y_test, preds)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    # Create a directory for saving evaluation results.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results/XGBoost_eval")
    os.makedirs(results_dir, exist_ok=True)
    
    # ------------------------------
    # 1. Confusion Matrix Plot
    # ------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ------------------------------
    # 2. Feature Importance Plot
    # ------------------------------
    feature_importance = booster.get_score(importance_type='weight')
    if feature_importance:
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(10, 6))
        plt.bar(list(sorted_importance.keys()), list(sorted_importance.values()))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance (weight)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        plt.close()
    else:
        print("Feature importance data is not available.")
    
    # ------------------------------
    # 3. ROC Curves & AUC (Per Class)
    # ------------------------------
    # Binarize true labels for ROC analysis (assumes 3 classes).
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], preds_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Per Class)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curves.png"))
    plt.close()
    
    # ------------------------------
    # 4. Precision-Recall Curves (Per Class)
    # ------------------------------
    pr_precision = dict()
    pr_recall = dict()
    for i in range(n_classes):
        pr_precision[i], pr_recall[i], _ = precision_recall_curve(y_test_bin[:, i], preds_prob[:, i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(pr_recall[i], pr_precision[i], label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Per Class)")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, "precision_recall_curves.png"))
    plt.close()
    
    # ------------------------------
    # 5. Summary Bar Plot for Metrics
    # ------------------------------
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [accuracy, precision, recall, f1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metric_names, y=metric_values)
    plt.ylim(0, 1)
    plt.title("Summary Metrics")
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(results_dir, "summary_metrics.png"))
    plt.close()
    
    # ------------------------------
    # 6. Save Evaluation Metrics to JSON File
    # ------------------------------
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": {f"class_{i}": roc_auc[i] for i in range(n_classes)},
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"XGBoost evaluation metrics and plots are saved to: {results_dir}")

if __name__ == "__main__":
    evaluate_xgboost()
