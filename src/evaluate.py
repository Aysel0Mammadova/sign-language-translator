import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from translator import ASLTranslator


def evaluate_all_modes(num_samples: int = 200):
    """
    Compares Pure Neural, Pure Rule-Based, and Hybrid Neuro-Symbolic models
    independently on the leak-free test split using BLEU-1 and BLEU-4.
    """
    test_path = os.path.join("data", "test_split.csv")
    test_df = pd.read_csv(test_path).dropna().head(num_samples)

    translator = ASLTranslator()
    smoothie = SmoothingFunction().method1

    results = {
        "neural": {"b1": [], "b4": []},
        "rule": {"b1": [], "b4": []},
        "hybrid": {"b1": [], "b4": []},
    }

    print(f"Evaluating 3 translation modes on {len(test_df)} unseen test sentences...\n")

    for _, row in test_df.iterrows():
        en_text = row["english"]
        true_gloss = row["gloss"].replace("<start>", "").replace("<end>", "").strip()
        ref_tokens = [true_gloss.split()]

        for mode in ["neural", "rule", "hybrid"]:
            pred_gloss = translator.translate(en_text, mode=mode)
            hyp_tokens = pred_gloss.split()

            b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            b4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

            results[mode]["b1"].append(b1)
            results[mode]["b4"].append(b4)

    print("=" * 60)
    print(f"{'Model Variant':<25} | {'Avg BLEU-1':<12} | {'Avg BLEU-4':<12}")
    print("-" * 60)
    for mode, label in [
        ("neural", "1. Pure Neural (Bi-LSTM)"),
        ("rule", "2. Pure Rule-Based (spaCy)"),
        ("hybrid", "3. Hybrid Neuro-Symbolic"),
    ]:
        avg_b1 = sum(results[mode]["b1"]) / len(results[mode]["b1"])
        avg_b4 = sum(results[mode]["b4"]) / len(results[mode]["b4"])
        print(f"{label:<25} | {avg_b1:<12.4f} | {avg_b4:<12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_all_modes(num_samples=200)