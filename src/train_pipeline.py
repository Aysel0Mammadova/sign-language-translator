import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    TextVectorization, Input, Embedding, LSTM, Bidirectional, Dense, Attention, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Configuration & Hyperparameters
DATA_PATH = os.path.join("data", "processed_asl.csv")
MODEL_DIR = "models"
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 16
EMBEDDING_DIM = 128
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 25


def build_seq2seq_attention_model(eng_vocab_size: int, asl_vocab_size: int) -> Model:
    """
    Constructs a Bidirectional Encoder-Decoder LSTM architecture with Luong Attention.
    """
    # --- BIDIRECTIONAL ENCODER ---
    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_emb = Embedding(
        input_dim=eng_vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_bilstm = Bidirectional(
        LSTM(LATENT_DIM, return_sequences=True, return_state=True),
        name="encoder_bilstm"
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_emb)

    # Concatenate forward and backward hidden/cell states
    state_h = Concatenate(name="concat_state_h")([forward_h, backward_h])
    state_c = Concatenate(name="concat_state_c")([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # --- DECODER (matches 2 * LATENT_DIM from Bidirectional Encoder) ---
    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_emb = Embedding(
        input_dim=asl_vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        LATENT_DIM * 2,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )
    decoder_lstm_outputs, _, _ = decoder_lstm(
        decoder_emb,
        initial_state=encoder_states
    )

    # --- LUONG ATTENTION ---
    attention_layer = Attention(name="attention_layer")
    context_vector = attention_layer([decoder_lstm_outputs, encoder_outputs])

    decoder_combined = Concatenate(axis=-1, name="concat_layer")(
        [decoder_lstm_outputs, context_vector]
    )

    decoder_dense = Dense(asl_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_combined)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_training():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna()
    print(f"Loaded {len(df)} sentence pairs from {DATA_PATH}")

    # Split into Train (80%), Validation (10%), Test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    test_df.to_csv(os.path.join("data", "test_split.csv"), index=False)
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit Vectorizers ONLY on Training Data
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
        dec_in = dec_full[:, :-1]
        dec_target = dec_full[:, 1:]
        return enc_in, dec_in, dec_target

    X_train_enc, X_train_dec, y_train = prepare_tensors(train_df)
    X_val_enc, X_val_dec, y_val = prepare_tensors(val_df)

    model = build_seq2seq_attention_model(len(eng_vocab), len(asl_vocab))

    weights_path = os.path.join(MODEL_DIR, "asl_attention.weights.h5")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    print("\nStarting training on Bidirectional Seq2Seq + Attention Model...")
    model.fit(
        [X_train_enc, X_train_dec],
        y_train,
        validation_data=([X_val_enc, X_val_dec], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"\nTraining complete! Best weights saved to '{weights_path}'.")


if __name__ == "__main__":
    run_training()