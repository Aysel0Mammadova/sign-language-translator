import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    TextVectorization, Input, Embedding, LSTM, Dense, Attention, Concatenate
)
from tensorflow.keras.models import Model

# 1. Expanded Parallel Dataset (Preparing structure for real CSV loading)
english_sentences = [
    "hello how are you",
    "i am going to school tomorrow",
    "she bought a new car yesterday",
    "what is your name",
    "i love learning artificial intelligence",
    "where do you live",
    "thank you very much",
    "i need help please",
    "he is reading a book now",
    "we will travel to baku next week",
    "do you understand sign language",
    "my friend works at a hospital",
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
    "<start> NOW IX-3P READ BOOK <end>",
    "<start> NEXT WEEK IX-1P-PLURAL TRAVEL BAKU <end>",
    "<start> IX-2P UNDERSTAND SIGN LANGUAGE <end>",
    "<start> IX-1P FRIEND WORK HOSPITAL <end>",
]

# 2. Hyperparameters
MAX_VOCAB_SIZE = 10000
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

# Create a reverse lookup dictionary (ID -> Word) for decoding predictions
asl_vocab = asl_vectorizer.get_vocabulary()
index_to_word = {idx: word for idx, word in enumerate(asl_vocab)}
word_to_index = {word: idx for idx, word in enumerate(asl_vocab)}

encoder_input_data = eng_vectorizer(english_sentences)
decoder_full_data = asl_vectorizer(asl_gloss_sequences)

decoder_input_data = decoder_full_data[:, :-1]
decoder_target_data = decoder_full_data[:, 1:]

# =====================================================================
# 4. BUILD SEQ2SEQ MODEL WITH ATTENTION MECHANISM
# =====================================================================

# --- ENCODER ---
encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
encoder_embedding = Embedding(
    input_dim=eng_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="encoder_embedding"
)(encoder_inputs)

# IMPORTANT: return_sequences=True is now needed so Attention can inspect EVERY input word
encoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    name="encoder_lstm"
)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# --- DECODER ---
decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
decoder_embedding = Embedding(
    input_dim=asl_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="decoder_embedding"
)(decoder_inputs)

decoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)
decoder_lstm_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

# --- ATTENTION LAYER (Luong-style Dot-Product Attention) ---
# Compares decoder outputs (queries) with all encoder outputs (values)
attention_layer = Attention(name="attention_layer")
context_vector = attention_layer([decoder_lstm_outputs, encoder_outputs])

# Concatenate the Attention context vector with the Decoder's own output
decoder_combined_context = Concatenate(axis=-1, name="concat_layer")(
    [decoder_lstm_outputs, context_vector]
)

# Final prediction layer
decoder_dense = Dense(asl_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================================================
# 5. INFERENCE FUNCTION (AUTOREGRESSIVE DECODING)
# =====================================================================
def translate_sentence(input_sentence: str) -> str:
    """
    Translates a raw English sentence into an ASL Gloss sequence
    word-by-word using the trained Attention Seq2Seq model.
    """
    # Vectorize the input English sentence
    enc_in = eng_vectorizer([input_sentence.lower().strip()])
    
    # Start the target sequence with the <start> token ID
    start_token_id = word_to_index["<start>"]
    end_token_id = word_to_index["<end>"]
    
    # Initialize decoder input array filled with zeros
    dec_in = np.zeros((1, MAX_SEQ_LENGTH - 1), dtype="int64")
    dec_in[0, 0] = start_token_id
    
    predicted_gloss = []
    
    for i in range(MAX_SEQ_LENGTH - 2):
        # Predict probabilities for all positions
        predictions = model.predict([enc_in, dec_in], verbose=0)
        
        # Get the word ID with the highest probability at the current time step 'i'
        predicted_id = np.argmax(predictions[0, i, :])
        
        # Stop if the model predicts <end> or padding (0)
        if predicted_id == end_token_id or predicted_id == 0:
            break
            
        predicted_word = index_to_word.get(predicted_id, "")
        predicted_gloss.append(predicted_word)
        
        # Feed the predicted word back into the decoder for the next step
        dec_in[0, i + 1] = predicted_id
        
    return " ".join(predicted_gloss)


if __name__ == "__main__":
    print("Training Seq2Seq + Attention Model...")
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=4,
        epochs=150,
        verbose=0
    )
    print("Training finished!\n")

    # Test real inference
    test_sentences = [
        "i am going to school tomorrow",
        "she bought a new car yesterday",
        "we will travel to baku next week",
        "do you understand sign language"
    ]
    
    for sent in test_sentences:
        translation = translate_sentence(sent)
        print(f"EN : {sent}")
        print(f"ASL: {translation}\n")