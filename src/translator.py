import os
import re
import json
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from model_builder import build_seq2seq_attention_model


class ASLTranslator:
    """
    Production Neuro-Symbolic ASL Translator supporting three inference modes:
      - 'neural': Pure Bidirectional Seq2Seq + Luong Attention output.
      - 'rule'  : Pure spaCy linguistic POS/lemma transformation.
      - 'hybrid': Combines neural generation with >=60% semantic overlap verification
                  and OOV/[UNK] preservation for downstream fingerspelling.
    """

    def __init__(self, model_dir: str = "models"):
        vocab_path = os.path.join(model_dir, "vocabularies.json")
        weights_path = os.path.join(model_dir, "asl_attention.weights.h5")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.eng_vocab = vocab_data["eng_vocab"]
        self.eng_vocab_set = set(self.eng_vocab)
        self.asl_vocab = vocab_data["asl_vocab"]
        self.max_seq_length = vocab_data.get("max_seq_length", 16)
        embedding_dim = vocab_data.get("embedding_dim", 128)
        latent_dim = vocab_data.get("latent_dim", 256)

        self.eng_vectorizer = TextVectorization(
            output_mode="int",
            output_sequence_length=self.max_seq_length,
            vocabulary=self.eng_vocab
        )

        self.index_to_word = {idx: word for idx, word in enumerate(self.asl_vocab)}
        self.word_to_index = {word: idx for idx, word in enumerate(self.asl_vocab)}

        self.model = build_seq2seq_attention_model(
            eng_vocab_size=len(self.eng_vocab),
            asl_vocab_size=len(self.asl_vocab),
            embedding_dim=embedding_dim,
            latent_dim=latent_dim
        )
        self.model.load_weights(weights_path)
        self.nlp = spacy.load("en_core_web_sm")

        # Unified ASLG-PC12 pronoun notation
        self.pronoun_map = {
            "i": "X-I",
            "me": "X-ME",
            "my": "X-MY",
            "you": "X-YOU",
            "your": "X-YOUR",
            "he": "X-HE",
            "she": "X-SHE",
            "it": "X-IT",
            "we": "X-WE",
            "us": "X-WE",
            "they": "X-THEY",
        }

    @staticmethod
    def _preprocess_input(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def translate_rule_based(self, cleaned_text: str) -> str:
        """Pure linguistic transformation using spaCy POS tagging and lemmatization."""
        doc = self.nlp(cleaned_text)
        time_tokens = []
        main_tokens = []

        for token in doc:
            if token.pos_ in {"DET", "PUNCT"} or token.lemma_.lower() == "be":
                continue
            if token.pos_ == "AUX" and token.lemma_.lower() in {"do", "have"}:
                continue

            lower_text = token.text.lower()
            gloss_word = self.pronoun_map.get(lower_text, token.lemma_.upper())

            if token.ent_type_ in {"DATE", "TIME"} or lower_text in {"tomorrow", "yesterday", "today", "now"}:
                time_tokens.append(gloss_word)
            else:
                main_tokens.append(gloss_word)

        return " ".join(time_tokens + main_tokens)

    def translate_neural(self, cleaned_text: str) -> str:
        """Fast autoregressive inference using direct model call (training=False)."""
        enc_in = self.eng_vectorizer([cleaned_text])
        start_token_id = self.word_to_index["<start>"]
        end_token_id = self.word_to_index["<end>"]

        dec_in = np.zeros((1, self.max_seq_length - 1), dtype="int64")
        dec_in[0, 0] = start_token_id

        neural_gloss = []
        for i in range(self.max_seq_length - 2):
            predictions = self.model([enc_in, tf.convert_to_tensor(dec_in)], training=False).numpy()
            predicted_id = int(np.argmax(predictions[0, i, :]))

            if predicted_id == end_token_id or predicted_id == 0:
                break

            predicted_word = self.index_to_word.get(predicted_id, "")
            if predicted_word and predicted_word != "[UNK]":
                neural_gloss.append(predicted_word)

            dec_in[0, i + 1] = predicted_id

        return " ".join(neural_gloss)

    def translate(self, english_sentence: str, mode: str = "hybrid") -> str:
        """
        Translates English text into ASL Gloss.
        Args:
            english_sentence: Raw input text.
            mode: 'neural', 'rule', or 'hybrid' (default).
        """
        cleaned_text = self._preprocess_input(english_sentence)
        if not cleaned_text:
            return ""

        if mode == "rule":
            return self.translate_rule_based(cleaned_text)

        neural_output = self.translate_neural(cleaned_text)
        if mode == "neural":
            return neural_output

        # Hybrid mode: Soft overlap threshold (>= 60%) + OOV preservation
        rule_output = self.translate_rule_based(cleaned_text)
        rule_tokens = rule_output.split()
        neural_tokens = neural_output.split()

        expected_content = {w for w in rule_tokens if not w.startswith("X-")}
        neural_content = {w for w in neural_tokens if not w.startswith("X-")}

        if not expected_content:
            return neural_output if neural_output else rule_output

        overlap_ratio = len(expected_content.intersection(neural_content)) / len(expected_content)

        # Preserve neural sequence if it captures >= 60% of core semantic concepts
        if overlap_ratio >= 0.60:
            # Ensure OOV words (e.g., proper nouns like BAKU) are not silently dropped
            oov_words = [
                tok.upper() for tok in cleaned_text.split()
                if tok not in self.eng_vocab_set and tok.upper() not in neural_tokens
            ]
            return " ".join(neural_tokens + oov_words)

        # Fall back to rule-based gloss only when neural prediction diverges heavily
        return rule_output