"""
Train, tune and save a XGBoost classifier.
"""
import os
import json
import random
import cv2
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna
from tqdm import tqdm
from xgboost.callback import TrainingCallback

# --------------------------
# Repeatability / Random Seed Setup
# --------------------------
def set_random_seed(seed=42):
    """
    Sets seeds for Python random module, NumPy, and sets an environment variable for hash seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# --------------------------
# Custom Callback for Progress Bar
# --------------------------
class TqdmCallback(TrainingCallback):
    """
    A custom XGBoost callback that displays a progress bar using tqdm.
    """
    def __init__(self, total_rounds):
        self.pbar = tqdm(total=total_rounds, desc="XGBoost Boosting Rounds", leave=True)
    
    def before_training(self, model):
        return model

    def after_training(self, model):
        self.pbar.close()
        return model

    def before_iteration(self, model, epoch, evals_log):
        return False

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        # Continue training.
        return False

# --------------------------
# Data Loading & Preprocessing
# --------------------------
def load_data(resize_shape=(32, 32)):
    """
    Loads and preprocesses training data for XGBoost.
    Each image is resized using OpenCV and then flattened.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../Data/preprocessed")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    
    # If labels are one-hot encoded, convert to indices.
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Resize images and flatten them
    X_train_resized = np.array([cv2.resize(img, resize_shape) for img in X_train])
    X_train_flat = X_train_resized.reshape(X_train_resized.shape[0], -1)
    return X_train_flat, y_train

# --------------------------
# Hyperparameter Tuning Objective Function using Optuna
# --------------------------
def objective(trial):
    # Ensure repeatability
    set_random_seed(42)
    
    # Load training data and split (80% train, 20% validation)
    X_train, y_train = load_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    
    # Suggest hyperparameters using updated suggestion functions.
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "eval_metric": "mlogloss",
        "seed": 42  # Fix seed for repeatability.
    }
    
    num_boost_round = 100
    early_stopping_rounds = 10
    evals = [(dtrain, "train"), (dval, "eval")]
    
    # Instantiate the progress bar callback.
    tqdm_cb = TqdmCallback(total_rounds=num_boost_round)
    
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[tqdm_cb],
        verbose_eval=False
    )
    
    # Use iteration_range to predict up to the best iteration.
    preds_prob = booster.predict(dval, iteration_range=(0, booster.best_iteration+1))
    preds = np.argmax(preds_prob, axis=1)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

# --------------------------
# Main Function: Hyperparameter Tuning & Final Training
# --------------------------
def main():
    set_random_seed(42)
    
    # Tune hyperparameters with Optuna.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    print("Best hyperparameters found:")
    print(best_params)
    
    # Load full training data.
    X_train, y_train = load_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    
    # Use the best hyperparameters (plus repeatability seed)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": best_params["max_depth"],
        "eta": best_params["eta"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "eval_metric": "mlogloss",
        "seed": 42
    }
    
    num_boost_round = 100
    early_stopping_rounds = 10
    evals = [(dtrain, "train"), (dval, "eval")]
    tqdm_cb = TqdmCallback(total_rounds=num_boost_round)
    
    print("Final training with the best hyperparameters...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[tqdm_cb],
        verbose_eval=True
    )
    
    preds_prob = booster.predict(dval, iteration_range=(0, booster.best_iteration+1))
    preds = np.argmax(preds_prob, axis=1)
    val_accuracy = accuracy_score(y_val, preds)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    # Save the final model and metadata.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../results/saved_model")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "xgboost_model.json")
    booster.save_model(model_path)
    
    meta_data = {
        "model_type": "XGBoost",
        "model_params": params,
        "resize_shape": (32, 32),
        "best_iteration": int(booster.best_iteration),
        "validation_accuracy": val_accuracy,
        "best_hyperparameters": best_params
    }
    
    params_path = os.path.join(script_dir, "../results", "xgboost_best_params.json")
    with open(params_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    
    print(f"XGBoost model saved to {model_path}")
    print(f"Model metadata saved to {params_path}")

if __name__ == "__main__":
    main()
