import pandas as pd

def load_human_scores(
    responses_path="data/human_responses.csv",
    message_id_col="message_id",
    score_col="score"
):
    """
    Expects human_responses.csv with columns:
    message_id, score   where score ∈ [0,10]
    """

    df = pd.read_csv(responses_path)

    # Normalize to probability scale
    df["human_prob"] = df[score_col] / 10.0

    # Aggregate per message
    agg = df.groupby(message_id_col).agg(
        human_mean_prob=("human_prob", "mean"),
        human_std_prob=("human_prob", "std"),
        num_raters=("human_prob", "count")
    ).reset_index()

    return agg

