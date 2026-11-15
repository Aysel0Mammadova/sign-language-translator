import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from translator import ASLTranslator


def evaluate_model(num_samples: int = 100):
    """
    Evaluates the trained ASLTranslator on unseen test data using BLEU-1 and BLEU-4 scores.
    """
    test_path = os.path.join("data", "test_split.csv")
    test_df = pd.read_csv(test_path).dropna().head(num_samples)

    translator = ASLTranslator()
    smoothie = SmoothingFunction().method1

    bleu1_scores = []
    bleu4_scores = []

    print(f"Evaluating model on {len(test_df)} unseen test sentences...\n")

    for idx, row in test_df.iterrows():
        en_text = row["english"]
        # Remove <start> and <end> tags from ground-truth gloss
        true_gloss = row["gloss"].replace("<start>", "").replace("<end>", "").strip()
        pred_gloss = translator.translate(en_text)

        ref_tokens = [true_gloss.split()]
        hyp_tokens = pred_gloss.split()

        # BLEU-1 measures unigram (word-level) accuracy
        b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        # BLEU-4 measures phrase/sequence-level accuracy
        b4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        bleu1_scores.append(b1)
        bleu4_scores.append(b4)

        if idx < 5:
            print(f"Sample {idx + 1}:")
            print(f"  English   : {en_text}")
            print(f"  True Gloss: {true_gloss}")
            print(f"  Pred Gloss: {pred_gloss}")
            print(f"  BLEU-1    : {b1:.4f}\n")

    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

    print("=" * 45)
    print(f"Evaluation Summary ({len(test_df)} Test Samples):")
    print(f"Average BLEU-1 Score (Word Accuracy) : {avg_bleu1:.4f}")
    print(f"Average BLEU-4 Score (Sequence Match): {avg_bleu4:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate_model(num_samples=100)