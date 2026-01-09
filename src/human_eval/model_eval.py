import pandas as pd
import joblib
import scipy.sparse as sp

def model_predict_on_human_set(
    test_path="data/human_test_set.csv"
):
    df = pd.read_csv(test_path)

    # Load trained artifacts
    vectorizer = joblib.load("results/vectorizer.pkl")
    lr = joblib.load("results/lr_model.pkl")

    # TF-IDF features
    X_tfidf = vectorizer.transform(df["text"])

    # Numeric features (MUST match training)
    df["text_length"] = df["text"].apply(len)
    df["num_exclam"] = df["text"].apply(lambda x: x.count("!"))
    df["num_digits"] = df["text"].apply(lambda x: sum(c.isdigit() for c in x))

    X_numeric = df[["text_length", "num_exclam", "num_digits"]].values
    X = sp.hstack([X_tfidf, X_numeric]).tocsr()

    # Model predictions
    df["model_prob"] = lr.predict_proba(X)[:, 1]
    df["model_label"] = (df["model_prob"] >= 0.5).astype(int)

    return df

