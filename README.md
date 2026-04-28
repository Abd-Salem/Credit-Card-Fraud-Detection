># 🔴 ***Credit-Card-Fraud-Detection***
---
![Fraud-Detection-Img](https://github.com/user-attachments/assets/7006358d-f330-4730-a6bb-cffe9a652cbc)


> A production-grade machine learning pipeline for detecting fraudulent credit card transactions — featuring classical ML ensembles, a custom PyTorch MLP with Focal Loss, aggressive resampling strategies, and YAML-driven configuration.
---
## 🚀 Live Demo

The trained model is served via a **FastAPI** — submit a transaction and get a fraud prediction with probability score in real time.

👉 [***Fraud Classifier Live Demo***](http://127.0.0.1:8000)


### Example Response

```json
{
  "prediction": 1,
  "probability": 0.94,
  "label": "Fraud"
}
```

> API built with **FastAPI** · Inference uses the best-performing model with tuned classification threshold
---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Stages](#project-stages)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Resampling Strategies](#resampling-strategies)
- [Evaluation](#evaluation)
- [Best Model](#best-model)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Results](#results)

---

## Overview

Credit card fraud is a needle-in-a-haystack problem — out of hundreds of thousands of transactions, only a handful are fraudulent. This project tackles that extreme class imbalance head-on through a multi-model, multi-resampling pipeline that goes beyond accuracy to optimize for what actually matters: **catching fraud while minimizing false alarms**.

---

## Dataset

The dataset is the well-known **[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** dataset, collected from European cardholders in September 2013.

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (~0.17%) |
| Features | V1–V28 (PCA-transformed), `Amount`, `Time` |
| Label | `Class` (0 = legitimate, 1 = fraud) |

---

## Project Stages

1. ***Data Resampling*** — Address class imbalance by generating balanced training sets via `SMOTE`, `RUS`, `SMOTETomek`, and `SMOTEENN`, then persist each variant for reproducibility.
2. ***Model Training & Evaluation*** — Train multiple classification algorithms across all resampling strategies and serialize evaluation metrics to `json` files for comparison.
3. ***Model Selection*** — Benchmark results across all experiments and retain the best-performing model based on `F1-score`.
4. ***Threshold Analysis*** — Stress-test the selected model under varying classification thresholds to find the optimal decision boundary.
5. ***Deployment*** — Expose the final model through a live `Fast API` for real-time fraud predictions.

---

## Project Structure
```
	Credit-Card-Fraud-Detection/
	│
	├── configs.yml                        # Central YAML configuration
	├── load_configs.py                    # Config loader
	│
	├── credit_fraud_train.py              # Training pipeline entry point
	├── credit_fraud_test.py               # Testing & threshold analysis entry point
	├── main.py                            # Main entry point
	│
	├── credit_fraud_utils_data.py         # Data loading & resampling utilities
	├── credit_fraud_utils_eval.py         # Evaluation metrics & scoring utilities
	├── credit_fraud_utils_helper.py       # General helper functions
	├── mlp_focal_loss.py                  # PyTorch MLP with Focal Loss (MLP_FL)
	│
	├── EDA.ipynb                          # Exploratory Data Analysis notebook
	│
	├── app/                               # FastAPI application
	│   ├── main.py                        # API routes & app setup
	│   ├── schemas/
	│   │   └── input_schemas.py           # Pydantic request/response schemas
	│   ├── service/
	│   │   └── predictor.py               # Inference logic & threshold application
	│   ├── static/
	│   │   └── style.css                  # Frontend styling
	│   └── templates/
	│       └── index.html                 # Demo UI template
	│
	├── split/                             # Raw stratified splits
	│   ├── train.csv
	│   ├── val.csv
	│   ├── train_val.csv
	│   └── test.csv
	│
	├── processed_data/sampled/            # Resampled training data (.npz per strategy)
	│   ├── smote/
	│   ├── enn/
	│   ├── smoteenn/
	│   ├── smotetomek/
	│   └── rus/
	│
	├── models/                            # Trained model artifacts
	│   ├── lr/                            # Logistic Regression
	│   ├── knn/                           # K-Nearest Neighbors
	│   ├── rf/                            # Random Forest
	│   ├── nn/                            # Neural Network (sklearn MLP)
	│   ├── nn-fl/                         # Neural Network with Focal Loss (PyTorch)
	│   ├── voting_1/                      # Voting Classifier variant 1
	│   ├── voting_2/                      # Voting Classifier variant 2
	│   └── voting_3/                      # Voting Classifier variant 3
	│       └── {strategy}/
	│           ├── model.pkl
	│           ├── model_eval_scores.json
	│           └── model_params.json
	│
	├── best_model/                        # Best selected model
	│   ├── best_model.pkl
	│   ├── best_model_metadata.json
	│   └── best_model_test_score.json
	│
	├── docs/evaluation/                   # PR curve plots per model & strategy
	│
	├── run_all_train_mode.sh              # Shell script to train all models
	├── run_all_eval_mode.sh               # Shell script to evaluate all models
	└── requirements.txt
```
---

## Pipeline Architecture

```
	Raw CSV
		└──► Resampling (train split) ──► .npz artifacts
	  		└──► Stratified Split (sampled train dataset)
	                    └──► sklearn Pipeline
	                              ├── RobustScaler or StandardScaler
	                              └── Estimator
	                                    └──► RandomizedSearchCV (AUPRC scorer)
	                                              └──► Best model ──► Threshold Tuning ──► Evaluation
```

All resampling happens **only on training data**. Validation and test sets remain untouched.

---

## Models

### Classical ML

| Model | Notes |
|---|---|
| `LogisticRegression` | Baseline; class-weight balanced |
| `RandomForestClassifier` | Strong ensemble baseline |
| `KNeighborsClassifier` | Distance-based, sensitive to scale |
| `MLPClassifier` (sklearn) | Shallow MLP for comparison |

### 🔥 Custom PyTorch MLP with Focal Loss (`MLP_FL`)

A fully sklearn-compatible neural network estimator — split into two classes to avoid PyTorch / sklearn multiple inheritance conflicts:

- **`_MLPNet(nn.Module)`** — pure PyTorch network: `Linear → ReLU → Dropout` blocks (BatchNorm deliberately excluded for numerical stability)

Training features:
- **Focal Loss** — focuses learning on hard-to-classify examples via `α` and `γ` parameters
- **Early stopping** — monitors validation loss
- `num_workers=0` for Windows DataLoader compatibility

```python
# Focal Loss formulation
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

### 🗳 VotingClassifier (Ensemble)

- A soft-voting ensemble built from multiple base estimators retrained using their best hyperparameters found during RandomizedSearchCV, combining `LR`, `KNN`, `RF`, and `MLP` variants for a stronger, more stable prediction.
---

## Resampling Strategies

Five resampling techniques are applied to the training fold only, with each variant saved as a `.npz` file:

| Strategy | Description |
|---|---|
| `SMOTE` | Synthetic Minority Over-sampling |
| `ENN` | Edited Nearest Neighbours (under-sampling) |
| `SMOTEENN` | SMOTE + ENN combined |
| `SMOTETomek` | SMOTE + Tomek Links combined |
| `RUS` | Random Under-Sampling |

---

## Evaluation

Standard accuracy is meaningless on this dataset. All evaluation is done with fraud-relevant metrics:

| Metric | Why |
|---|---|
| **AUPRC** | Primary metric — area under Precision-Recall curve; robust to imbalance |
| **F2-score** | Weights recall twice as heavily as precision — missing fraud is more costly |
| **Threshold tuning** | `precision_recall_curve` used to find the classification threshold that maximizes F2 on training data |

> ROC-AUC is intentionally deprioritized — it is overly optimistic on imbalanced datasets.

---
## Best Model

After benchmarking all models across every resampling strategy, the best performing model is a **soft-voting ensemble (VotingClassifier)** combining `RF`, `NN`, and `KNN` — each retrained on **SMOTE** with their optimal hyperparameters — and saved to `best_model/`.

### Sub-model Configuration

| Model | Sampling | Scaler | Key Params |
|---|---|---|---|
| Random Forest | SMOTE | RobustScaler | `n_estimators=600`, `max_depth=20`, `class_weight=balanced` |
| Neural Network | SMOTE | StandardScaler | `hidden_layers=(128,64,32)`, `activation=relu`, `batch_size=128` |
| KNN | SMOTE | RobustScaler | `n_neighbors=9`, `weights=distance`, `metric=manhattan` |

Voting weights: `[RF=2, NN=1, KNN=1]`

### Test Set Results

| Threshold | F1-Score (Fraud) |
|---|---|
| 0.65 | 84.58% |

| Metric | Score |
|---|---|
| AUPRC | 85.88% |
| Best F1-Score (th=0.65) | 84.5% |

> Full test evaluation details are logged in `best_model/best_model_test_score.json`

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/Abd-Salem/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Create and activate conda environment
conda create -n fraud-detection python=3.13
conda activate fraud-detection

# Install dependencies
python -m pip install -r requirements.txt
```

**Required packages include:**
`scikit-learn`, `imbalanced-learn`, `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `pyyaml`, `fastapi`, `uvicorn`, `pydantic`, `jinja2`, `requests`

> 💡 On Windows, if your Conda path contains spaces, always use `python -m pip install` instead of `pip install`.

---

## Configuration

All experiment parameters live in `configs.yml`:

```yaml
random_state: 34
seed: 12

dataset:
  unprocessed:
    train: 'split/train.csv'
    val: 'split/val.csv'
    train_val: 'split/train_val.csv'
    test: 'split/test.csv'
  sampled:
    dir: 'processed_data/sampled'
    # one dir per strategy: rus, enn, smote, smoteenn, smotetomek

features:
  input: ['Time', 'V1', ..., 'V28', 'Amount']
  target: 'Class'

preprocessing:
  scaler: ['standard', 'robust']

sampling:
  techniques: ['rus', 'enn', 'smote', 'smoteenn', 'smotetomek']
  params:
    sampling_strategy: 'auto'
    smote_kn: 5
    enn_nn: 3

models:
  logistic_regression:
    params:
      solver: ['lbfgs', 'newton-cg', 'liblinear']
      max_iter: [100, 500, 800, 1000, 2000]
      class_weight: ['balanced', {1: 5}, {1: 10}, {1: 20}]
  random_forest:
    n_iter: 25
    params:
      n_estimators: [300, 400, 600]
      max_depth: [15, 20, 25]
      max_features: ['sqrt']
  neural_network:
    n_iter: 30
    params:
      hidden_layers: [[32], [64], [64,32], [128,64], [128,64,32]]
      activation: ['relu', 'tanh']
      batch_size: [32, 64, 128]
  neural_network_fl:
    params:
      alpha: [0.25, 0.5, 0.75]
      gamma: [2.0, 3.0]
      hidden_layers: [[32], [64,32], [128,32], [128,64,32]]
      epochs: [50, 100, 200]
      patience: 30
  knn_classifier:
    params:
      k: [3, 5, 7, 9]
      weights: ['uniform', 'distance']
      metric: ['manhattan', 'euclidean', 'minkowski']

evaluation:
  scoring: 'average_precision'   # AUPRC
  beta: 2                        # F2-score
  cv_folds: 5
```

---

## Results

> Full per-model, per-strategy results are logged to `reports/` after training.

Key findings:
- **SMOTE** , **SMOTETomek** and **SMOTEENN** consistently outperform pure over- or under-sampling
- **Focal Loss MLP** achieves competitive AUPRC with strong recall on hard fraud cases
- Threshold tuning yields meaningful F2 gains over the default 0.5 cutoff across all models
- **VotingClassifier** (soft voting) provides the most stable performance across validation folds
