import os
import random
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer


# Hand-crafted conversational benchmark (never present in ASLG-PC12 parliamentary corpus)
MANUAL_CONVERSATIONAL_BENCHMARK = [
    ("I will travel tomorrow", "TOMORROW X-I TRAVEL"),
    ("What is your name", "X-YOU NAME WHAT"),
    ("Where do you live", "X-YOU LIVE WHERE"),
    ("Where do you work", "X-YOU WORK WHERE"),
    ("I live in Baku", "X-I LIVE BAKU"),
    ("I am a doctor", "X-I DOCTOR"),
    ("I am a student", "X-I STUDENT"),
    ("She is a teacher", "X-SHE TEACHER"),
    ("He is reading a book now", "NOW X-HE READ BOOK"),
    ("I need help please", "X-I NEED HELP PLEASE"),
    ("Thank you very much", "THANK-YOU VERY-MUCH"),
    ("Yes I understand", "YES X-I UNDERSTAND"),
    ("No I do not understand", "NO X-I NOT UNDERSTAND"),
    ("I am going to school tomorrow", "TOMORROW X-I GO SCHOOL"),
    ("She bought a new car yesterday", "YESTERDAY X-SHE BUY NEW CAR"),
    ("Good morning", "GOOD MORNING"),
    ("Good evening", "GOOD EVENING"),
    ("My friend works at a hospital", "X-I FRIEND WORK HOSPITAL"),
    ("I study at university", "X-I STUDY UNIVERSITY"),
    ("I am learning sign language", "X-I LEARN SIGN LANGUAGE"),
]


def compute_wer(ref_tokens: list, hyp_tokens: list) -> float:
    """Computes standard Word Error Rate (Levenshtein distance / reference length)."""
    r_len, h_len = len(ref_tokens), len(hyp_tokens)
    if r_len == 0:
        return 0.0 if h_len == 0 else 1.0

    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[r_len][h_len] / r_len


def evaluate_benchmark(pairs: list, translator: ASLTranslator, visualizer: ASLSignVisualizer, title: str):
    smoothie = SmoothingFunction().method1
    print(f"\n{'=' * 75}")
    print(f"BENCHMARK: {title} ({len(pairs)} sentences)")
    print(f"{'=' * 75}")
    print(f"{'Model Variant':<26} | {'Corpus BLEU':<11} | {'WER (↓)':<9} | {'Exact Match':<11} | {'Sign Coverage'}")
    print("-" * 75)

    for mode, label in [
        ("neural", "1. Pure Neural (Bi-LSTM)"),
        ("rule", "2. Pure Rule-Based (spaCy)"),
        ("hybrid", "3. Hybrid Neuro-Symbolic"),
    ]:
        references = []
        hypotheses = []
        wers = []
        exact_matches = 0
        mapped_tokens = 0
        total_tokens = 0

        for en_text, true_gloss in pairs:
            pred_gloss = translator.translate(en_text, mode=mode)
            ref_toks = true_gloss.split()
            hyp_toks = pred_gloss.split()

            references.append([ref_toks])
            hypotheses.append(hyp_toks)
            wers.append(compute_wer(ref_toks, hyp_toks))

            if ref_toks == hyp_toks:
                exact_matches += 1

            # Measure real visual sign coverage (excluding missing tokens)
            visuals = visualizer.gloss_to_visuals(pred_gloss, allow_fingerspelling=False)
            for v in visuals:
                total_tokens += 1
                if v["media_type"] != "missing":
                    mapped_tokens += 1

        c_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
        avg_wer = sum(wers) / len(wers)
        exact_pct = (exact_matches / len(pairs)) * 100
        coverage_pct = (mapped_tokens / total_tokens * 100) if total_tokens > 0 else 0.0

        print(f"{label:<26} | {c_bleu:<11.4f} | {avg_wer:<9.4f} | {exact_pct:>9.1f}% | {coverage_pct:>10.1f}%")


if __name__ == "__main__":
    translator = ASLTranslator()
    visualizer = ASLSignVisualizer()

    # 1. Evaluate on Random Sample of ASLG-PC12 Test Split
    test_path = os.path.join("data", "test_split.csv")
    test_df = pd.read_csv(test_path).dropna().sample(n=300, random_state=42)
    aslg_pairs = [
        (row["english"], row["gloss"].replace("<start>", "").replace("<end>", "").strip())
        for _, row in test_df.iterrows()
    ]
    evaluate_benchmark(aslg_pairs, translator, visualizer, "ASLG-PC12 Random Test Split")

    # 2. Evaluate on Hand-Crafted Conversational Benchmark
    evaluate_benchmark(MANUAL_CONVERSATIONAL_BENCHMARK, translator, visualizer, "Conversational Real-World Set")