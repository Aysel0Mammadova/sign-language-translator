from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Attention, Concatenate
)
from tensorflow.keras.models import Model


def build_seq2seq_attention_model(
    eng_vocab_size: int,
    asl_vocab_size: int,
    embedding_dim: int = 128,
    latent_dim: int = 256
) -> Model:
    """
    Constructs a Bidirectional Encoder-Decoder LSTM architecture with Luong Attention.
    Decoupled from training scripts so inference modules can load weights cleanly.
    """
    # Bidirectional Encoder
    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_emb = Embedding(
        input_dim=eng_vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_bilstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, return_state=True),
        name="encoder_bilstm"
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_emb)

    state_h = Concatenate(name="concat_state_h")([forward_h, backward_h])
    state_c = Concatenate(name="concat_state_c")([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder (hidden size = 2 * latent_dim)
    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
    decoder_emb = Embedding(
        input_dim=asl_vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim * 2,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )
    decoder_lstm_outputs, _, _ = decoder_lstm(
        decoder_emb,
        initial_state=encoder_states
    )

    # Luong Dot-Product Attention
    attention_layer = Attention(name="attention_layer")
    context_vector = attention_layer([decoder_lstm_outputs, encoder_outputs])

    decoder_combined = Concatenate(axis=-1, name="concat_layer")(
        [decoder_lstm_outputs, context_vector]
    )

    decoder_outputs = Dense(asl_vocab_size, activation="softmax", name="decoder_dense")(
        decoder_combined
    )

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model