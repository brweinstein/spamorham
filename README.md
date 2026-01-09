# Spam Detection with Naive Bayes & Logistic Regression

This repository contains a spam detection pipeline using **Python**, **scikit-learn**, and **TF-IDF features**. It includes model training, evaluation, and cross-validation analysis. This uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data) from Kaggle.

## Project Structure
```bash
├── human_experiment.md
├── README.md
├── requirements.txt
├── data/
│   ├── spam.csv
│   └── human_test_set.csv
├── results/
└── src/
    ├── features.py
    ├── train.py
    ├── evaluate.py
    ├── cross_validate.py
    └── human_eval/
        ├── human_data.py
        ├── model_eval.py
        ├── compare_human_model.py
        └── __init__.py
```

## Setup

```bash
git clone <repo_url>
cd spam

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage
```bash
python src/features.py
python src/train.py
python src/evaluate.py
python src/cross_validate.py

# Compare model to human judgements
python src/human_eval/compare_human_model.py
```

## Features

- TF-IDF vectors of the text (max 2000 features)

- Numeric features:
    - Text length
    - Number of exclamation marks
    - Number of digits

## Models

- Naive Bayes
- Logistical Regression

## Results
Cross-validation metrics:

| Model               | Accuracy       | ROC-AUC       |
|--------------------|----------------|---------------|
| Naive Bayes         | $0.971 \pm 0.008$ | $0.948 \pm 0.018$ |
| Logistic Regression | $0.978 \pm 0.006$ | $0.994 \pm 0.003$ |

## Human vs Model Experiment
In addition to automated evaluation, this project includes a human experiment designed to compare human intuition against machine predictions on a random set of spam and ham messages.

Participants rated messages on a 0–10 confidence scale, where higher values indicate greater confidence that a message is spam. Human scores are normalized to probabilities and directly compared against model outputs using accuracy, ROC-AUC, and error analysis.

See `human_experiment.md` for full methodology, dataset construction, and experimental details.

## Disclaimer

This project is intended **solely for educational and research purposes**.  
It is designed to compare human intuition against machine learning models on a spam classification task.

The models in this repository are **not intended for commercial use, deployment, or real-world decision-making systems**.
