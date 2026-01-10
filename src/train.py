from features import load_data, extract_features
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
import joblib

def train_models(test_size=0.2, random_state=42):
    df = load_data()
    X, y, vectorizer = extract_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train models
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    # Evaluate on test set
    y_prob_nb = nb.predict_proba(X_test)[:,1]
    y_pred_nb = nb.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:,1]
    y_pred_lr = lr.predict(X_test)

    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Naive Bayes ROC-AUC:", roc_auc_score(y_test, y_prob_nb))
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

    Path("results/").mkdir(exist_ok=True)

    # Save models and vectorizer for evaluation
    joblib.dump(nb, 'results/nb_model.pkl')
    joblib.dump(lr, 'results/lr_model.pkl')
    joblib.dump(vectorizer, 'results/vectorizer.pkl')
    joblib.dump((X_test, y_test), 'results/test_split.pkl')

    return nb, lr, vectorizer, X_test, y_test, y_prob_nb, y_prob_lr

if __name__ == "__main__":
    train_models()

