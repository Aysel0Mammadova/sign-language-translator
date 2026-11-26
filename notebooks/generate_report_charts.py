import os
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from evaluate import evaluate_benchmark, UNSEEN_CONVERSATIONAL_BENCHMARK
from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer

sns.set_theme(style="whitegrid")
os.makedirs("assets", exist_ok=True)


def plot_dataset_statistics():
    """Generates discrete text EDA charts for ASLG-PC12 and saves to 'assets/dataset_eda.png'."""
    data_path = os.path.join("data", "processed_asl.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path).dropna()
    df["en_word_count"] = df["english"].apply(lambda x: len(str(x).split()))
    df["gloss_word_count"] = df["gloss"].apply(lambda x: max(1, len(str(x).split()) - 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["en_word_count"], discrete=True, color="#2563EB",
                 label="English Input", ax=axes[0], alpha=0.55)
    sns.histplot(df["gloss_word_count"], discrete=True, color="#10B981",
                 label="ASL Gloss Output", ax=axes[0], alpha=0.55)
    axes[0].set_title("Sentence Length Distribution (filtered to max 14 words)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Tokens")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    all_gloss_tokens = [
        tok for g in df["gloss"] for tok in str(g).split()
        if tok not in {"<start>", "<end>"}
    ]
    top_tokens = pd.Series(all_gloss_tokens).value_counts().head(12)
    sns.barplot(x=top_tokens.values, y=top_tokens.index, hue=top_tokens.index,
                palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("Top 12 Most Frequent ASL Gloss Tokens in Corpus", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Occurrences")
    axes[1].set_ylabel("ASL Gloss Token")

    plt.tight_layout()
    chart_path = os.path.join("assets", "dataset_eda.png")
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved Text EDA visualization to '{chart_path}'")


def plot_multimodal_eda():
    """Generates sample ASL handshape grid and video FPS/duration distribution (Requirement #2)."""
    alpha_dir = os.path.join("data", "signs", "alphabet")
    vid_dir = os.path.join("data", "signs", "videos")

    fig = plt.figure(figsize=(14, 5))

    # Left subplot: 2x4 grid of sample ASL alphabet hand-signs
    sample_letters = ["A", "B", "C", "D", "H", "L", "O", "Y"]
    for idx, letter in enumerate(sample_letters):
        ax = fig.add_subplot(2, 8, (idx // 4) * 8 + (idx % 4) + 1)
        img_path = os.path.join(alpha_dir, f"{letter}.png")
        if os.path.exists(img_path):
            ax.imshow(Image.open(img_path))
        ax.set_title(letter, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Right subplot: Video clip duration distribution
    ax_vid = fig.add_subplot(1, 2, 2)
    durations = []
    if os.path.isdir(vid_dir):
        for f in os.listdir(vid_dir):
            if f.endswith(".mp4"):
                cap = cv2.VideoCapture(os.path.join(vid_dir, f))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if fps and fps > 0 and frames > 0:
                    durations.append(round(frames / fps, 2))

    if durations:
        sns.histplot(durations, bins=10, color="#8B5CF6", ax=ax_vid)
        ax_vid.set_title("Dynamic Sign Video Duration Distribution (sec)", fontsize=12, fontweight="bold")
        ax_vid.set_xlabel("Duration (seconds)")
        ax_vid.set_ylabel("Video Count")

    plt.tight_layout()
    out_path = os.path.join("assets", "multimodal_eda.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Multimodal EDA visualization to '{out_path}'")


if __name__ == "__main__":
    print("1. Generating Text & Multimodal EDA Charts...")
    plot_dataset_statistics()
    plot_multimodal_eda()

    print("\n2. Running Leak-Free Quantitative Evaluation...")
    translator = ASLTranslator()
    visualizer = ASLSignVisualizer()

    test_df = pd.read_csv(os.path.join("data", "test_split.csv")).dropna().sample(n=300, random_state=42)
    test_pairs = [
        (r["english"], r["gloss"].replace("<start>", "").replace("<end>", "").strip())
        for _, r in test_df.iterrows()
    ]
    evaluate_benchmark(test_pairs, translator, visualizer, "Full ASLG-PC12 Test Split")
    evaluate_benchmark(UNSEEN_CONVERSATIONAL_BENCHMARK, translator, visualizer, "100% Unseen Conversational Benchmark")