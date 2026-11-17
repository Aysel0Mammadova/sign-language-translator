import os
import re
import pandas as pd
from datasets import load_dataset


def clean_text(text: str, is_gloss: bool = False) -> str:
    """Normalizes whitespace and strips punctuation while preserving ASL gloss hyphens."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if is_gloss:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9\-\s]", "", text)
    else:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.strip()


def build_clean_corpus(max_samples: int = 35000, max_words: int = 14):
    """
    Downloads and deduplicates the ASLG-PC12 parallel corpus.
    Domain oversampling is deferred strictly until AFTER train/val/test splitting
    in train_pipeline.py to prevent data leakage.
    """
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "processed_asl.csv")

    print("Downloading ASLG-PC12 dataset from Hugging Face...")
    dataset = load_dataset("achrafothman/aslg_pc12", split="train")

    records = []
    for item in dataset:
        en_clean = clean_text(item.get("text", ""), is_gloss=False)
        gloss_clean = clean_text(item.get("gloss", ""), is_gloss=True)

        if not en_clean or not gloss_clean:
            continue
        if len(en_clean.split()) > max_words or len(gloss_clean.split()) > max_words:
            continue

        records.append({
            "english": en_clean,
            "gloss": f"<start> {gloss_clean} <end>"
        })

        if len(records) >= max_samples:
            break

    df_clean = pd.DataFrame(records).drop_duplicates(subset=["english"]).reset_index(drop=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved {len(df_clean)} unique sentence pairs to '{output_path}'.")


if __name__ == "__main__":
    build_clean_corpus(max_samples=35000, max_words=14)