import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_human_vs_model():
    df = pd.read_csv('results/human_model_comparison.csv')
    
    question_ids = df['question_id']
    human_means = df['human_mean']
    nb_scores = df['nb_score']
    lr_scores = df['lr_score']
    
    x = np.arange(len(question_ids))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bars1 = ax.bar(x - width, human_means, width, label='Human Mean', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x, nb_scores, width, label='Naive Bayes', alpha=0.8, color='#A23B72')
    bars3 = ax.bar(x + width, lr_scores, width, label='Logistic Regression', alpha=0.8, color='#F18F01')
    
    ax.set_xlabel('Question ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spam Score (0-10)', fontsize=12, fontweight='bold')
    ax.set_title('Human vs Model Spam Predictions Across 30 Test Messages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(question_ids)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 10.5)
    
    plt.tight_layout()
    plt.savefig('results/human_vs_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved to results/human_vs_model_comparison.png")

if __name__ == "__main__":
    plot_human_vs_model()
