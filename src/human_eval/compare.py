import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from pathlib import Path

def prepare_text_for_prediction(texts, vectorizer):
    X_tfidf = vectorizer.transform(texts)
    
    text_lengths = texts.apply(len).values.reshape(-1, 1)
    num_exclams = texts.apply(lambda x: x.count('!')).values.reshape(-1, 1)
    num_digits = texts.apply(lambda x: sum(c.isdigit() for c in x)).values.reshape(-1, 1)
    
    X_numeric = np.hstack([text_lengths, num_exclams, num_digits])
    X = sp.hstack([X_tfidf, X_numeric])
    
    return X

def compare_human_vs_model():
    human_df = pd.read_csv('data/human_responses.csv')
    
    test_df = pd.read_csv('data/human_test_set.csv')
    test_30 = test_df.head(30).copy()
    
    nb = joblib.load('results/nb_model.pkl')
    lr = joblib.load('results/lr_model.pkl')
    vectorizer = joblib.load('results/vectorizer.pkl')
    
    X_test_30 = prepare_text_for_prediction(test_30['text'], vectorizer)
    
    nb_probs = nb.predict_proba(X_test_30)[:, 1] * 10
    lr_probs = lr. predict_proba(X_test_30)[:, 1] * 10
    
    human_cols = [str(i) for i in range(1, 31)]
    human_responses = human_df[human_cols]
    
    human_responses = human_responses.apply(pd.to_numeric, errors='coerce')
    
    human_means = np.array(human_responses.mean(axis=0), dtype=float)
    human_stds = np.array(human_responses.std(axis=0), dtype=float)
    
    nb_probs = np.asarray(nb_probs, dtype=float)
    lr_probs = np.asarray(lr_probs, dtype=float)
    
    comparison = pd.DataFrame({
        'question_id': range(1, 31),
        'text': test_30['text']. values,
        'true_label': test_30['spam_label'].values,
        'human_mean': human_means,
        'human_std': human_stds,
        'nb_score': nb_probs,
        'lr_score': lr_probs,
        'nb_error': np.abs(nb_probs - human_means),
        'lr_error': np. abs(lr_probs - human_means),
    })
    
    nb_corr = np.corrcoef(human_means, nb_probs)[0, 1]
    lr_corr = np.corrcoef(human_means, lr_probs)[0, 1]
    
    nb_mae = comparison['nb_error'].mean()
    lr_mae = comparison['lr_error'].mean()
    
    print("="*60)
    print("HUMAN vs MODEL COMPARISON (30 Test Questions)")
    print("="*60)
    print(f"\nNumber of human raters:  {len(human_df)}")
    print(f"\nCorrelations with human mean scores:")
    print(f"  Naive Bayes:         {nb_corr:.4f}")
    print(f"  Logistic Regression: {lr_corr:.4f}")
    print(f"\nMean Absolute Error (0-10 scale):")
    print(f"  Naive Bayes:          {nb_mae:.4f}")
    print(f"  Logistic Regression: {lr_mae:.4f}")
    
    print("\n" + "="*60)
    print("TOP 5 BIGGEST DISAGREEMENTS (Naive Bayes vs Human)")
    print("="*60)
    top_nb_errors = comparison.nlargest(5, 'nb_error')[
        ['question_id', 'human_mean', 'nb_score', 'nb_error', 'text']
    ]
    for _, row in top_nb_errors.iterrows():
        print(f"\nQ{row['question_id']}: Human={row['human_mean']:.1f}, NB={row['nb_score']:.1f}, Diff={row['nb_error']:.1f}")
        print(f"  Text: {row['text'][:80]}...")
    
    print("\n" + "="*60)
    print("TOP 5 BIGGEST DISAGREEMENTS (Logistic Regression vs Human)")
    print("="*60)
    top_lr_errors = comparison.nlargest(5, 'lr_error')[
        ['question_id', 'human_mean', 'lr_score', 'lr_error', 'text']
    ]
    for _, row in top_lr_errors.iterrows():
        print(f"\nQ{row['question_id']}: Human={row['human_mean']:.1f}, LR={row['lr_score']:.1f}, Diff={row['lr_error']:.1f}")
        print(f"  Text: {row['text'][:80]}...")
    
    Path("results/").mkdir(exist_ok=True)
    comparison.to_csv('results/human_model_comparison.csv', index=False)
    print("\n" + "="*60)
    print("Detailed comparison saved to:  results/human_model_comparison. csv")
    print("="*60)
    
    return comparison

if __name__ == "__main__":
    compare_human_vs_model()
