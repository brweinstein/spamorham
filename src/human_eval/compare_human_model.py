from sklearn.metrics import accuracy_score, roc_auc_score
from .human_data import load_human_scores
from .model_eval import model_predict_on_human_set

def run_comparison():
    # Load model predictions
    model_df = model_predict_on_human_set()

    # Load aggregated human responses
    human_df = load_human_scores()

    # Merge on message id
    df = model_df.merge(
        human_df,
        left_on="id",
        right_on="message_id",
        how="inner"
    )

    # Convert human probabilities to binary labels
    df["human_label"] = (df["human_mean_prob"] >= 0.5).astype(int)

    print("\n=== Accuracy ===")
    print("Human Accuracy:",
          accuracy_score(df["true_label"], df["human_label"]))
    print("Model Accuracy:",
          accuracy_score(df["true_label"], df["model_label"]))

    print("\n=== ROC-AUC ===")
    print("Human ROC-AUC:",
          roc_auc_score(df["true_label"], df["human_mean_prob"]))
    print("Model ROC-AUC:",
          roc_auc_score(df["true_label"], df["model_prob"]))

    # Error analysis
    df["human_fp"] = (df["human_label"] == 1) & (df["true_label"] == 0)
    df["human_fn"] = (df["human_label"] == 0) & (df["true_label"] == 1)
    df["model_fp"] = (df["model_label"] == 1) & (df["true_label"] == 0)
    df["model_fn"] = (df["model_label"] == 0) & (df["true_label"] == 1)

    print("\n=== Error Counts ===")
    print("Human False Positives:", df["human_fp"].sum())
    print("Human False Negatives:", df["human_fn"].sum())
    print("Model False Positives:", df["model_fp"].sum())
    print("Model False Negatives:", df["model_fn"].sum())

    # Save for notebook analysis
    df.to_csv("results/human_vs_model_comparison.csv", index=False)
    print("\nSaved results to results/human_vs_model_comparison.csv")

if __name__ == "__main__":
    run_comparison()
