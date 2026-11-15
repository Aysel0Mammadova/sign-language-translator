import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path to evaluate our actual translator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from evaluate import evaluate_model

# Set professional plot styling
sns.set_theme(style="whitegrid")
os.makedirs("assets", exist_ok=True)


def plot_dataset_statistics():
    """
    Generates Exploratory Data Analysis (EDA) charts for the processed ASLG-PC12 dataset
    and saves them into the 'assets/' folder for documentation.
    """
    data_path = os.path.join("data", "processed_asl.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path).dropna()

    # Calculate word counts for English and ASL Gloss
    df["en_word_count"] = df["english"].apply(lambda x: len(str(x).split()))
    # Subtract 2 for <start> and <end> tokens in gloss
    df["gloss_word_count"] = df["gloss"].apply(lambda x: max(1, len(str(x).split()) - 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Sentence Length Distribution Comparison
    sns.histplot(df["en_word_count"], bins=14, color="#2563EB", label="English Input", kde=True, ax=axes[0], alpha=0.5)
    sns.histplot(df["gloss_word_count"], bins=14, color="#10B981", label="ASL Gloss Output", kde=True, ax=axes[0], alpha=0.5)
    axes[0].set_title("Sentence Length Distribution: English vs. ASL Gloss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Tokens")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Plot 2: Top 12 Most Frequent ASL Gloss Tokens
    all_gloss_tokens = [
        tok for g in df["gloss"] for tok in str(g).split()
        if tok not in {"<start>", "<end>"}
    ]
    top_tokens = pd.Series(all_gloss_tokens).value_counts().head(12)

    sns.barplot(x=top_tokens.values, y=top_tokens.index, palette="viridis", ax=axes[1])
    axes[1].set_title("Top 12 Most Frequent ASL Gloss Tokens in Corpus", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Occurrences")
    axes[1].set_ylabel("ASL Gloss Token")

    plt.tight_layout()
    chart_path = os.path.join("assets", "dataset_eda.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Saved EDA visualization to '{chart_path}'")


if __name__ == "__main__":
    print("1. Generating Dataset EDA Charts...")
    plot_dataset_statistics()

    print("\n2. Running Quantitative BLEU Evaluation on Test Split...")
    evaluate_model(num_samples=200)