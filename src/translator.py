import os
import re
import json
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from train_pipeline import build_seq2seq_attention_model


class ASLTranslator:
    """
    Hybrid Neuro-Symbolic ASL Translator:
    1. Uses the trained Bidirectional Seq2Seq + Attention neural model for learned patterns.
    2. Uses spaCy POS/Lemmatization guardrails to prevent neural hallucinations on rare words
       and enforce ASL grammar (Time-Topic-Comment, dropping articles & copula verbs).
    """

    def __init__(self, model_dir: str = "models"):
        vocab_path = os.path.join(model_dir, "vocabularies.json")
        weights_path = os.path.join(model_dir, "asl_attention.weights.h5")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.eng_vocab = vocab_data["eng_vocab"]
        self.asl_vocab = vocab_data["asl_vocab"]
        self.max_seq_length = vocab_data["max_seq_length"]

        self.eng_vectorizer = TextVectorization(
            output_mode="int",
            output_sequence_length=self.max_seq_length,
            vocabulary=self.eng_vocab
        )

        self.index_to_word = {idx: word for idx, word in enumerate(self.asl_vocab)}
        self.word_to_index = {word: idx for idx, word in enumerate(self.asl_vocab)}

        self.model = build_seq2seq_attention_model(
            eng_vocab_size=len(self.eng_vocab),
            asl_vocab_size=len(self.asl_vocab)
        )
        self.model.load_weights(weights_path)
        self.nlp = spacy.load("en_core_web_sm")

        # Pronoun mapping matching ASLG-PC12 standard
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

    def _preprocess_input(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _linguistic_gloss_fallback(self, cleaned_text: str) -> str:
        """
        Generates a grammatically accurate ASL Gloss sequence using spaCy
        dependency & POS analysis (Time-Topic-Comment order, lemma extraction).
        """
        doc = self.nlp(cleaned_text)
        time_tokens = []
        main_tokens = []

        for token in doc:
            # Skip English articles (a, an, the), copula 'be' verbs, and punctuation
            if token.pos_ in {"DET", "PUNCT"} or token.lemma_.lower() == "be":
                continue
            if token.pos_ == "AUX" and token.lemma_.lower() in {"do", "have"}:
                continue

            lower_text = token.text.lower()
            if lower_text in self.pronoun_map:
                gloss_word = self.pronoun_map[lower_text]
            else:
                gloss_word = token.lemma_.upper()

            if token.ent_type_ in {"DATE", "TIME"} or lower_text in {"tomorrow", "yesterday", "today", "now"}:
                time_tokens.append(gloss_word)
            else:
                main_tokens.append(gloss_word)

        return " ".join(time_tokens + main_tokens)

    def translate(self, english_sentence: str) -> str:
        cleaned_text = self._preprocess_input(english_sentence)
        if not cleaned_text:
            return ""

        # Step 1: Run Neural Seq2Seq + Attention Prediction
        enc_in = self.eng_vectorizer([cleaned_text])
        start_token_id = self.word_to_index["<start>"]
        end_token_id = self.word_to_index["<end>"]

        dec_in = np.zeros((1, self.max_seq_length - 1), dtype="int64")
        dec_in[0, 0] = start_token_id

        neural_gloss = []
        for i in range(self.max_seq_length - 2):
            predictions = self.model.predict([enc_in, dec_in], verbose=0)
            predicted_id = int(np.argmax(predictions[0, i, :]))

            if predicted_id == end_token_id or predicted_id == 0:
                break

            predicted_word = self.index_to_word.get(predicted_id, "")
            if predicted_word and predicted_word != "[UNK]":
                neural_gloss.append(predicted_word)

            dec_in[0, i + 1] = predicted_id

        # Step 2: Verify semantic alignment against input content lemmas
        # Prevents rare-word hallucinations (e.g., translating 'doctor' as 'BLOCKER')
        linguistic_gloss = self._linguistic_gloss_fallback(cleaned_text)
        expected_content_words = {
            w for w in linguistic_gloss.split() if not w.startswith("X-")
        }
        neural_content_words = {
            w for w in neural_gloss if not w.startswith("X-")
        }

        # If neural output hallucinates unrelated words not grounded in the input,
        # use the linguistically verified ASL Gloss structure
        if not expected_content_words.issubset(neural_content_words):
            return linguistic_gloss

        return " ".join(neural_gloss)