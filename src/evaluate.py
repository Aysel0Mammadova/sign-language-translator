import os
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer


# 100% Unseen Conversational Benchmark:
# None of these sentences appear in train_pipeline.py's conversational augmentation set.
UNSEEN_CONVERSATIONAL_BENCHMARK = [
    ("Where do they live", "X-THEY LIVE WHERE"),
    ("We are happy today", "TODAY X-WE HAPPY"),
    ("My brother is a nurse", "X-I BROTHER NURSE"),
    ("She will read a book tomorrow", "TOMORROW X-SHE READ BOOK"),
    ("He works at a new school", "X-HE WORK NEW SCHOOL"),
    ("Do you understand this problem", "X-YOU UNDERSTAND THIS PROBLEM"),
    ("I bought a book yesterday", "YESTERDAY X-I BUY BOOK"),
    ("They need help today", "TODAY X-THEY NEED HELP"),
    ("Where is the hospital", "HOSPITAL WHERE"),
    ("My friend is learning English", "X-I FRIEND LEARN ENGLISH"),
    ("We will go to the university", "X-WE WILL GO UNIVERSITY"),
    ("She does not like coffee", "X-SHE NOT LIKE COFFEE"),
    ("What do you need", "X-YOU NEED WHAT"),
    ("He is a good teacher", "X-HE GOOD TEACHER"),
    ("I live in Ganja", "X-I LIVE GANJA"),
    ("Tomorrow we will study together", "TOMORROW X-WE STUDY TOGETHER"),
    ("Your friend helped me yesterday", "YESTERDAY X-YOU FRIEND HELP X-I"),
    ("Why are you sad", "X-YOU SAD WHY"),
    ("I want to learn sign language", "X-I WANT LEARN SIGN LANGUAGE"),
    ("Good afternoon my friend", "GOOD AFTERNOON X-I FRIEND"),
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
    print(f"\n{'=' * 78}")
    print(f"BENCHMARK: {title} ({len(pairs)} sentences)")
    print(f"{'=' * 78}")
    print(f"{'Model Variant':<26} | {'Corpus BLEU':<11} | {'WER (↓)':<9} | {'Exact Match':<11} | {'Media Coverage'}")
    print("-" * 78)

    for mode, label in [
        ("neural", "1. Pure Neural (Bi-LSTM)"),
        ("rule", "2. Pure Rule-Based (spaCy)"),
        ("hybrid", "3. Hybrid Neuro-Symbolic"),
    ]:
        references = []
        hypotheses = []
        wers = []
        exact_matches = 0
        media_tokens = 0
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

            # Count ONLY authentic video/image files (exclude text-only instruction cards)
            visuals = visualizer.gloss_to_visuals(pred_gloss, allow_fingerspelling=False)
            for v in visuals:
                total_tokens += 1
                if v["media_type"] in {"video", "gif", "image"}:
                    media_tokens += 1

        c_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
        avg_wer = sum(wers) / len(wers)
        exact_pct = (exact_matches / len(pairs)) * 100
        coverage_pct = (media_tokens / total_tokens * 100) if total_tokens > 0 else 0.0

        print(f"{label:<26} | {c_bleu:<11.4f} | {avg_wer:<9.4f} | {exact_pct:>9.1f}% | {coverage_pct:>11.1f}%")


if __name__ == "__main__":
    translator = ASLTranslator()
    visualizer = ASLSignVisualizer()

    # 1. Full ASLG-PC12 Test Split
    test_path = os.path.join("data", "test_split.csv")
    test_df = pd.read_csv(test_path).dropna()
    aslg_pairs = [
        (row["english"], row["gloss"].replace("<start>", "").replace("<end>", "").strip())
        for _, row in test_df.iterrows()
    ]
    evaluate_benchmark(aslg_pairs, translator, visualizer, "Full ASLG-PC12 Test Split")

    # 2. 100% Unseen Hand-Crafted Conversational Benchmark
    evaluate_benchmark(UNSEEN_CONVERSATIONAL_BENCHMARK, translator, visualizer, "Unseen Conversational Set")