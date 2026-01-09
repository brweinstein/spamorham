import numpy as np
import pandas as pd
from features import load_data, extract_features
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def cross_validate_models(n_splits=5, random_state=42):
    df = load_data()
    X, y, _ = extract_features(df)

    X = X.tocsr()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    nb_accs, nb_rocs = [], []
    lr_accs, lr_rocs = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        y_prob_nb = nb.predict_proba(X_test)[:,1]

        nb_accs.append(accuracy_score(y_test, y_pred_nb))
        nb_rocs.append(roc_auc_score(y_test, y_prob_nb))

        # Logistic Regression
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:,1]

        lr_accs.append(accuracy_score(y_test, y_pred_lr))
        lr_rocs.append(roc_auc_score(y_test, y_prob_lr))

        print(f"Fold {fold} done.")

    # Summary
    print("\n=== Cross-Validation Results ===")
    print(f"Naive Bayes Accuracy: {np.mean(nb_accs):.4f} ± {np.std(nb_accs):.4f}")
    print(f"Naive Bayes ROC-AUC: {np.mean(nb_rocs):.4f} ± {np.std(nb_rocs):.4f}")
    print(f"Logistic Regression Accuracy: {np.mean(lr_accs):.4f} ± {np.std(lr_accs):.4f}")
    print(f"Logistic Regression ROC-AUC: {np.mean(lr_rocs):.4f} ± {np.std(lr_rocs):.4f}")

if __name__ == "__main__":
    cross_validate_models()

