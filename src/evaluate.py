import pandas as pd
import joblib

def save_predictions():
    # Load models and test set
    nb = joblib.load('results/nb_model.pkl')
    lr = joblib.load('results/lr_model.pkl')
    X_test, y_test = joblib.load('results/test_split.pkl')

    # Predict probabilities
    y_prob_nb = nb.predict_proba(X_test)[:,1]
    y_prob_lr = lr.predict_proba(X_test)[:,1]

    # Save to CSV
    df_eval = pd.DataFrame({
        'true_label': y_test,
        'nb_prob': y_prob_nb,
        'lr_prob': y_prob_lr
    })

    df_eval.to_csv('results/model_probs.csv', index=False)
    print("Predictions saved to results/model_probs.csv")

if __name__ == "__main__":
    save_predictions()

