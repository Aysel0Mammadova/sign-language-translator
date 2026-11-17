import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_builder import build_seq2seq_attention_model

DATA_PATH = os.path.join("data", "processed_asl.csv")
MODEL_DIR = "models"
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 16
EMBEDDING_DIM = 128
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 25


def get_train_only_conversational_pairs() -> pd.DataFrame:
    """
    Conversational English -> ASL Gloss pairs injected strictly into the training split
    after splitting to avoid any data leakage into validation or test sets.
    """
    pairs = [
        ("hello how are you", "HELLO HOW X-YOU"),
        ("hello", "HELLO"),
        ("good morning", "GOOD MORNING"),
        ("good evening", "GOOD EVENING"),
        ("what is your name", "X-YOU NAME WHAT"),
        ("i am a student", "X-I STUDENT"),
        ("i am a doctor", "X-I DOCTOR"),
        ("she is a student", "X-SHE STUDENT"),
        ("he is a teacher", "X-HE TEACHER"),
        ("i study at university", "X-I STUDY UNIVERSITY"),
        ("i am learning sign language", "X-I LEARN SIGN LANGUAGE"),
        ("where do you live", "X-YOU LIVE WHERE"),
        ("i live in baku", "X-I LIVE BAKU"),
        ("i am going to school tomorrow", "TOMORROW X-I GO SCHOOL"),
        ("i will travel tomorrow", "TOMORROW X-I WILL TRAVEL"),
        ("where do you work", "X-YOU WORK WHERE"),
        ("i need help please", "X-I NEED HELP PLEASE"),
        ("thank you very much", "THANK-YOU VERY-MUCH"),
        ("you are welcome", "X-YOU WELCOME"),
        ("yes i understand", "YES X-I UNDERSTAND"),
        ("no i do not understand", "NO X-I NOT UNDERSTAND"),
    ]
    rows = [
        {"english": en, "gloss": f"<start> {gloss} <end>"}
        for en, gloss in pairs * 15
    ]
    return pd.DataFrame(rows)


def run_training():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna().drop_duplicates(subset=["english"])
    print(f"Loaded {len(df)} unique pairs from {DATA_PATH}")

    # 1. Perform strict split FIRST (zero data leakage)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save untouched test split for honest evaluation
    test_df.to_csv(os.path.join("data", "test_split.csv"), index=False)

    # 2. Augment ONLY the training set with conversational domain examples
    conv_df = get_train_only_conversational_pairs()
    train_df = pd.concat([train_df, conv_df], ignore_index=True).sample(frac=1.0, random_state=42)

    print(f"Leak-free Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    eng_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQ_LENGTH
    )
    asl_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        standardize=None,
        output_mode="int",
        output_sequence_length=MAX_SEQ_LENGTH
    )

    eng_vectorizer.adapt(train_df["english"].values)
    asl_vectorizer.adapt(train_df["gloss"].values)

    eng_vocab = eng_vectorizer.get_vocabulary()
    asl_vocab = asl_vectorizer.get_vocabulary()

    vocab_data = {
        "eng_vocab": eng_vocab,
        "asl_vocab": asl_vocab,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "latent_dim": LATENT_DIM
    }
    with open(os.path.join(MODEL_DIR, "vocabularies.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def prepare_tensors(dataframe):
        enc_in = eng_vectorizer(dataframe["english"].values)
        dec_full = asl_vectorizer(dataframe["gloss"].values)
        return enc_in, dec_full[:, :-1], dec_full[:, 1:]

    X_train_enc, X_train_dec, y_train = prepare_tensors(train_df)
    X_val_enc, X_val_dec, y_val = prepare_tensors(val_df)

    model = build_seq2seq_attention_model(
        eng_vocab_size=len(eng_vocab),
        asl_vocab_size=len(asl_vocab),
        embedding_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM
    )

    weights_path = os.path.join(MODEL_DIR, "asl_attention.weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    ]

    model.fit(
        [X_train_enc, X_train_dec],
        y_train,
        validation_data=([X_val_enc, X_val_dec], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"Training complete! Weights saved to '{weights_path}'.")


if __name__ == "__main__":
    run_training()