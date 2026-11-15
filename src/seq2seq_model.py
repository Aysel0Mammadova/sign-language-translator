import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# 1. Sample Parallel Dataset (English -> ASL Gloss)
english_sentences = [
    "hello how are you",
    "i am going to school tomorrow",
    "she bought a new car yesterday",
    "what is your name",
    "i love learning artificial intelligence",
    "where do you live",
    "thank you very much",
    "i need help please",
]

asl_gloss_sequences = [
    "<start> HELLO HOW IX-2P <end>",
    "<start> TOMORROW IX-1P GO SCHOOL <end>",
    "<start> YESTERDAY IX-3P BUY NEW CAR <end>",
    "<start> IX-2P NAME WHAT <end>",
    "<start> IX-1P LOVE LEARN ARTIFICIAL INTELLIGENCE <end>",
    "<start> IX-2P LIVE WHERE <end>",
    "<start> THANK-YOU VERY-MUCH <end>",
    "<start> IX-1P NEED HELP PLEASE <end>",
]

# 2. Hyperparameters
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LENGTH = 15
EMBEDDING_DIM = 128
LATENT_DIM = 256

# 3. Text Vectorization
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

eng_vectorizer.adapt(english_sentences)
asl_vectorizer.adapt(asl_gloss_sequences)

eng_vocab_size = len(eng_vectorizer.get_vocabulary())
asl_vocab_size = len(asl_vectorizer.get_vocabulary())

encoder_input_data = eng_vectorizer(english_sentences)
decoder_full_data = asl_vectorizer(asl_gloss_sequences)

decoder_input_data = decoder_full_data[:, :-1]
decoder_target_data = decoder_full_data[:, 1:]

# --- ENCODER ---
encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
encoder_embedding = Embedding(
    input_dim=eng_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,  # Ignore padded zeros during training
    name="encoder_embedding"
)(encoder_inputs)

# return_state=True outputs the hidden state (state_h) and cell state (state_c)
encoder_lstm = LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]  # Context vector summarizing the input sentence

# --- DECODER ---
decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
decoder_embedding_layer = Embedding(
    input_dim=asl_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="decoder_embedding"
)
decoder_embedding = decoder_embedding_layer(decoder_inputs)

# return_sequences=True is required because we need predictions at EVERY time step
decoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states  # Inject Encoder's memory into Decoder
)

# Final classification layer predicting the probability of each ASL word
decoder_dense = Dense(asl_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# Combine Encoder and Decoder into a single trainable Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 5. Compile the Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

if __name__ == "__main__":
    model.summary()
    print("\nStarting training (Sanity Check on sample data)...")
    
    # Train the model for 150 epochs on our tiny dataset to verify learning capacity
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=4,
        epochs=150,
        verbose=0  # Suppress per-epoch logs for clean output
    )
    
    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    print(f"Training completed! Final Loss: {final_loss:.4f} | Final Accuracy: {final_acc:.4f}")