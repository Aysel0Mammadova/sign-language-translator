import os
import re
import pandas as pd
from datasets import load_dataset


def clean_text(text: str, is_gloss: bool = False) -> str:
    """
    Cleans raw text or gloss strings by removing unwanted characters
    and normalizing whitespace.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if is_gloss:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9\-\s]", "", text)
    else:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.strip()


def get_everyday_conversation_pairs() -> list:
    """
    Provides curated everyday English -> ASL Gloss pairs matching ASLG-PC12 notation
    (e.g., X-I, X-YOU, X-HE) to fix domain shift from parliamentary text.
    """
    pairs = [
        ("hello how are you", "HELLO HOW X-YOU"),
        ("hello", "HELLO"),
        ("hi", "HELLO"),
        ("good morning", "GOOD MORNING"),
        ("good evening", "GOOD EVENING"),
        ("what is your name", "X-YOU NAME WHAT"),
        ("my name is student", "X-I NAME STUDENT"),
        ("i am a student", "X-I STUDENT"),
        ("she is a student", "X-SHE STUDENT"),
        ("he is a teacher", "X-HE TEACHER"),
        ("i study at university", "X-I STUDY UNIVERSITY"),
        ("i am learning sign language", "X-I LEARN SIGN LANGUAGE"),
        ("i love artificial intelligence", "X-I LOVE ARTIFICIAL INTELLIGENCE"),
        ("where do you live", "X-YOU LIVE WHERE"),
        ("i live in baku", "X-I LIVE BAKU"),
        ("i am going to school tomorrow", "TOMORROW X-I GO SCHOOL"),
        ("she bought a new car yesterday", "YESTERDAY X-SHE BUY NEW CAR"),
        ("i need help please", "X-I NEED HELP PLEASE"),
        ("can you help me", "X-YOU HELP X-ME"),
        ("thank you very much", "THANK-YOU VERY-MUCH"),
        ("you are welcome", "X-YOU WELCOME"),
        ("yes i understand", "YES X-I UNDERSTAND"),
        ("no i do not understand", "NO X-I NOT UNDERSTAND"),
        ("he is reading a book now", "NOW X-HE READ BOOK"),
        ("my friend works at a hospital", "X-I FRIEND WORK HOSPITAL"),
        ("we will travel tomorrow", "TOMORROW X-WE TRAVEL"),
    ]
    # Oversample everyday pairs (repeat 15x) so the neural network learns them reliably
    # alongside the large parliamentary dataset
    oversampled = []
    for en, gloss in pairs * 15:
        oversampled.append({
            "english": clean_text(en, is_gloss=False),
            "gloss": f"<start> {clean_text(gloss, is_gloss=True)} <end>"
        })
    return oversampled


def build_dataset(max_samples: int = 35000, max_words: int = 14):
    """
    Downloads the ASLG-PC12 parallel corpus, merges it with everyday conversational pairs,
    and saves the clean dataset to 'data/processed_asl.csv'.
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

    df_corpus = pd.DataFrame(records).drop_duplicates()
    df_everyday = pd.DataFrame(get_everyday_conversation_pairs())

    # Combine and shuffle the dataset
    df_final = pd.concat([df_corpus, df_everyday], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_final.to_csv(output_path, index=False)
    print(f"Dataset saved to '{output_path}' with {len(df_final)} total training pairs.")


if __name__ == "__main__":
    build_dataset(max_samples=35000, max_words=14)