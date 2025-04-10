# Meat Freshness Classification Project
## Overview

This project focuses on classifying meat images into three categories: Fresh, Half-Fresh, and Spoiled. The objective is to enhance food safety by minimizing the risk of serving spoiled meat through a robust machine learning pipeline.

## Dataset

- **Meat Freshness Image Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset)

## Models and Methods

- **Baseline CNN**: A three-layer convolutional neural network tuned with Optuna.
- **Modified CNN with Screening**: Combines LIME and RandomForest to mask unimportant pixels before training.
- **Vision Transformer (ViT)**: A patch-based transformer model with hyperparameter tuning.
- **XGBoost**: Gradient-boosted tree classifier using flattened 32×32 images.
- **K-Means Clustering**: PCA + k=3 clustering to validate natural class structure.

## Instructionsd
1. **Install Dependencies**  
   ```Follow the instructions in requirements.txt to bring your system up to speed in terms of pytorch, CUDA, CUDNN and other packages. You will not be able to run this code in a reasonable time frame without an NVIDIA GPU. Even with this acceleration it still takes multiple hours to tune the models.
   ```

2. **Prepare Data**  
   - Download and unzip the dataset into `data/Meat_Freshness_Data/`.
   - Run preprocessing:  
     ```bash
     python src/data_loading.py
     ```
#If running from the start execute the full pipeline after preparing data.
3. **Train and Evaluate Models**  
   - **Baseline CNN**:  
     ```bash
     python src/train_CNN.py  
     python src/evaluate_CNN.py
     ```s
   - **XGBoost**:  
     ```bash
     python src/train_XGBoost.py  
     python src/evaluate_XGBoost.py
     ```
   - **ViT**:  
     ```bash
     python src/train_VIT.py  
     python src/evaluate_VIT.py
     ```
   - **Modified CNN**:  
     ```bash
     python src/CNN_Modified_Input.py  
     python src/evaluate_CNN_Modified_Input.py
     ```
   - **Full Pipeline**:  
     ```bash
     python src/run_all.py
     ```
4. **Evaluate Findings and Data Analysis**
notebooks/Exploratory_Data_Analysis.ipynb
notebooks/Analysis_Final_Results.ipynb
## Key Findings

- **Modified CNN**: Best balance of accuracy and safety, reducing misclassification of spoiled meat by over 50%.
- **ViT**: Highest raw accuracy but requires a larger model size (>100 MB).
- **XGBoost**: Compact (~50 MB) with good performance for resource-constrained environments.
- **K-Means Clustering**: Validated three natural groups in the data (~75% clustering accuracy).

## Recommendation

Deploy the **Modified CNN** in production for its interpretability-driven masking and low misclassification risk.
This project presents a comprehensive machine learning pipeline for classifying the freshness of meat samples. The goal is to automatically distinguish between Fresh, Half-Fresh, and Spoiled meat images to enhance food safety and quality control. Our approach combines several modeling techniques, interpretability methods, and evaluation strategies to ensure robust performance and consumer safety.

## Project Structure

- **Data Loading & Preprocessing**  
  The `data_loading.py` script loads images from the Meat Freshness dataset, resizes them to a consistent shape (224×224), normalizes pixel values, and performs validation checks. Processed data are saved as NumPy arrays for efficient downstream access.

- **Exploratory Data Analysis (EDA)**  
  The `Exploratory_Data_Analysis.ipynb` notebook provides an in-depth look at the dataset statistics, label distribution, and quality checks. This analysis helped inform subsequent modeling decisions.

- **Model Training**  
  Multiple model architectures are explored:
  - **Standard CNN:** Trained using PyTorch with hyperparameter tuning via Optuna. (See `train_CNN.py`)
  - **Modified CNN with Screening Masks:** This approach leverages interpretability methods (LIME and RandomForest) to generate a combined importance mask. This mask is applied to the input images to remove uninformative noise before CNN training. (See `CNN_Modified_Input.py`)
  - **Vision Transformer (ViT):** A simplified transformer model (SimpleViT) is tuned and trained for image classification. (See `train_VIT.py`)
  - **XGBoost Classifier:** A classical gradient boosting method is also implemented with hyperparameter tuning and is used as a benchmark. (See `train_XGBoost.py`)

- **Model Evaluation**  
  Each model has its dedicated evaluation script that:
  - Loads test data and applies the necessary preprocessing transformations.
  - Computes a full suite of evaluation metrics including accuracy, precision, recall, F1-score, ROC/AUC, and confusion matrices.
  - Generates visualizations (confusion matrices, ROC curves, precision-recall curves, summary bar charts) that are saved for further analysis.  
  Evaluation scripts include `evaluate_CNN.py`, `evaluate_CNN_Modified_Input.py`, `evaluate_VIT.py`, and `evaluate_XGBoost.py`.

- **Interpretability & Feature Screening**  
  The modified CNN pipeline incorporates a screening algorithm that:
  - Uses LIME to generate per-pixel importance masks for a sample of images.
  - Extracts feature importance using a RandomForest classifier.
  - Aggregates and thresholds these importance maps to create a binary mask that suppresses noise.
  - This screening enhances the CNN’s focus on critical regions, reducing dangerous misclassifications (e.g., labeling spoiled meat as fresh).  
  See `CNN_Modified_Input.py` and the corresponding analysis in `Analysis_Final_Results.ipynb`.

- **Analysis & Comparison**  
  The `Analysis_Final_Results.ipynb` notebook consolidates outputs from all training and evaluation scripts. It includes:
  - A detailed comparison of the models based on accuracy, F1 score, model size (storage), and the risk of misclassifying spoiled meat.
  - Visualizations that use dual-axis plots to compare performance metrics and model storage requirements.
  - A K‑Means clustering analysis to confirm the natural grouping in the data.
  - A summary of the screening algorithm and interpretability outputs.

## Repository Contents

- **/Data/**  
  Contains raw and preprocessed data (images and labels).

- **/results/**  
  Stores trained model checkpoints, meta-data files, and evaluation outputs (plots and JSON files).

- **/src/**  
  Includes all Python scripts:
  - `data_loading.py`
  - `train_CNN.py`
  - `train_XGBoost.py`
  - `train_VIT.py`
  - `CNN_Modified_Input.py`
  - `evaluate_CNN.py`
  - `evaluate_CNN_Modified_Input.py`
  - `evaluate_VIT.py`
  - `evaluate_XGBoost.py`
  - `run_all.py`

- **Notebooks/**  
  Contains Jupyter notebooks for exploratory data analysis (`Exploratory_Data_Analysis.ipynb`) and final analysis (`Analysis_Final_Results.ipynb`). 

## Conclusion and Recommendations

THe best model accuracy was achieved with the Baseline CNN at 91.35%, however, this came at the cost of an expensive model with a reisk of misclassication of 11.40% against consumers which is the not the best result on the list. The modified CNN scored worse than any other model on the list due to the screen being designed as binary. Further manipulations may have produced a model with better outcomes, however, that will be in future steps. ViT did best when it came correctly identifying spoiled meat as spoiled with only 6.14% misidentified in that way, however, the overall model accuracy was almost 10% worse. The model is slightly smaller in size but not by much. The XGBoost method produce a method with similar accuracy to the VIT, a much smalller size but a misclassification risk of almsot 24%. That means implementing such a model is unlikely to offer automated quality control abilities here. Based on the constraints of a manufacturing site, most likely, the XGBoost model will be reccomended due its smaller size. Better tuning and a more powerful machine fo training overcome its accuracy challenges.

**Future Steps:**
- **Fix Modified CNN** modify the screening algorithim to preserve more of the data. Tune it iteratively to assist in pre-screening incoming image data before the model is being validated. 
- **Explore lightweight transformer architectures** to merge ViT accuracy with reduced model size.
- **Continuously monitor** misclassification rates on new data and update the screening process as necessary.d

## Acknowledgments

This project builds upon advanced interpretability and hyperparameter tuning techniques. Special thanks to the contributors of LIME, RandomForest, PyTorch, and XGBoost for making these tools available to the community.
