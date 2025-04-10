"""
Locates and transforms the existing image data and labels into a single dataframe
Run checks on the data to make sure that everything has been setup correctly.
Modify dataframes to be readily usable for model creation
"""
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

def load_and_preprocess_data():
    """
    Loads and preprocesses image data, performs dataset validation checks, 
    and returns the processed data.

    Returns:
        tuple: A tuple containing the training and testing data (X_train, y_train, X_test, y_test).
    """
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths to the data directories
    data_dir = os.path.join(script_dir, "../Data")
    train_path = os.path.join(data_dir, "Meat_Freshness_Data", "train")
    test_path = os.path.join(data_dir, "Meat_Freshness_Data", "valid")

    def load_labels(folder):
        labels_file = os.path.join(folder, "_classes.csv")
        df = pd.read_csv(labels_file)
        return df

    def process_images(folder, labels_df):
        images, labels = [], []
        for _, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0]):
            img_path = os.path.join(folder, row["filename"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                images.append(img)

                label = row.iloc[1:].tolist()
                labels.append(label)

        return np.array(images), np.array(labels)

    train_labels = load_labels(train_path)
    test_labels = load_labels(test_path)

    X_train, y_train = process_images(train_path, train_labels)
    X_test, y_test = process_images(test_path, test_labels)

    # Perform validation checks
    perform_validation_checks(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test

def perform_validation_checks(X_train, y_train, X_test, y_test):
    """
    Performs validation checks on the dataset and prints summaries.

    Args:
        X_train, y_train, X_test, y_test: Processed data arrays.
    """
    # Dataset Summary
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    image_shape = X_train.shape[1:]  # Assuming all images have the same shape
    num_features = np.prod(image_shape)  # Total number of pixels (features)

    print(f"Dataset Summary:")
    print(f"  - Number of training samples: {num_train_samples}")
    print(f"  - Number of testing samples: {num_test_samples}")
    print(f"  - Image shape: {image_shape}")
    print(f"  - Number of features (pixels): {num_features}")

    # Dataset Description
    print("\nDataset Description:")
    print("""
    This dataset consists of images of meat samples categorized into three freshness levels:
    Fresh, Half-Fresh, and Spoiled. The goal is to build a machine learning model that can
    automatically classify the freshness of meat based on its visual features. This is important
    for quality control in food production and retail, helping to ensure food safety and reduce waste.
    """)

    print("\nFeature and Target Explanation:")
    print("  - Features:")
    print("    - Image Pixels (continuous): Each pixel in the image represents a feature. The pixel values represent the intensity of red, green, and blue color channels.")
    print("  - Targets:")
    print("    - Freshness Category (categorical, determined by binary columns): Fresh, Half-Fresh, and Spoiled.")

    # Counts/Frequencies for Categorical Features
    train_label_counts = pd.DataFrame(y_train, columns=["Fresh", "Half-Fresh", "Spoiled"]).apply(pd.value_counts)
    test_label_counts = pd.DataFrame(y_test, columns=["Fresh", "Half-Fresh", "Spoiled"]).apply(pd.value_counts)

    print("\nCounts/Frequencies for Categorical Features (Training Data):")
    print(train_label_counts)
    print("\nCounts/Frequencies for Categorical Features (Testing Data):")
    print(test_label_counts)

    # Check for Missing Values
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    train_missing = np.isnan(X_train_flat).sum(axis=0)  # Missing values in training data
    test_missing = np.isnan(X_test_flat).sum(axis=0)   # Missing values in testing data

    if train_missing.any() or test_missing.any():
        print("\nMissing Values Found!")
        print(f"Missing in Training Data: {train_missing}")
        print(f"Missing in Testing Data: {test_missing}")
    else:
        print("\nNo missing values detected in the dataset.")

def save_data(X_train, y_train, X_test, y_test):
    """
    Saves the preprocessed data to NumPy files (.npy) using absolute paths.

    Args:
        X_train, y_train, X_test, y_test: The preprocessed data arrays.
    """
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the output directory
    output_dir = os.path.join(script_dir, "../Data/preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    save_data(X_train, y_train, X_test, y_test)