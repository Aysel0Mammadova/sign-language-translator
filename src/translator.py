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
    Unified Neuro-Symbolic ASL Translator:
    - Enforces consistent ASLG-PC12 pronoun notation (X-I, X-YOU, X-HE, etc.).
    - Runs spaCy on raw cased text to preserve NER (Baku, dates) and POS accuracy.
    - Applies ASL syntactic rules: Time-Topic-Comment, WH-word movement to end,
      and omission of 'WILL' when explicit future time markers exist.
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

        # Unified pronoun mapping matching training pairs (e.g., your -> X-YOU)
        self.pronoun_map = {
            "i": "X-I",
            "me": "X-I",
            "my": "X-I",
            "you": "X-YOU",
            "your": "X-YOU",
            "he": "X-HE",
            "him": "X-HE",
            "his": "X-HE",
            "she": "X-SHE",
            "her": "X-SHE",
            "it": "X-IT",
            "we": "X-WE",
            "us": "X-WE",
            "our": "X-WE",
            "they": "X-THEY",
            "them": "X-THEY",
        }
        self.wh_words = {"WHAT", "WHERE", "WHO", "WHY", "HOW", "WHICH", "WHEN"}
        self.time_words = {"TOMORROW", "YESTERDAY", "TODAY", "NOW", "SOON", "LATER"}

    @staticmethod
    def _clean_for_neural(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def translate_rule_based(self, raw_sentence: str) -> tuple[str, set]:
        """
        Transforms raw cased English text into ASL Gloss using spaCy NER/POS.
        Returns (gloss_string, proper_nouns_and_oov_set).
        """
        doc = self.nlp(raw_sentence.strip())
        time_tokens = []
        main_tokens = []
        wh_tokens = []
        spellable_tokens = set()

        has_explicit_time = any(
            tok.text.upper() in self.time_words or tok.ent_type_ in {"DATE", "TIME"}
            for tok in doc
        )

        for token in doc:
            if token.is_punct or token.is_space or token.pos_ == "DET":
                continue
            if token.lemma_.lower() == "be":
                continue
            if token.pos_ == "AUX" and token.lemma_.lower() in {"do", "have"}:
                continue
            # In ASL, 'WILL' is redundant when an explicit future time word is present
            if token.lemma_.lower() == "will" and has_explicit_time:
                continue

            lower_text = token.text.lower()
            if lower_text in self.pronoun_map:
                gloss_word = self.pronoun_map[lower_text]
            else:
                gloss_word = token.lemma_.upper()

            # Track genuine Proper Nouns (e.g., Baku) or OOV words eligible for fingerspelling
            if token.pos_ == "PROPN" or lower_text not in self.eng_vocab_set:
                spellable_tokens.add(gloss_word)

            if gloss_word in self.wh_words:
                wh_tokens.append(gloss_word)
            elif gloss_word in self.time_words or token.ent_type_ in {"DATE", "TIME"}:
                time_tokens.append(gloss_word)
            else:
                main_tokens.append(gloss_word)

        ordered = time_tokens + main_tokens + wh_tokens
        return " ".join(ordered), spellable_tokens

    def translate_neural(self, raw_sentence: str) -> str:
        cleaned = self._clean_for_neural(raw_sentence)
        if not cleaned:
            return ""

        enc_in = self.eng_vectorizer([cleaned])
        start_id = self.word_to_index["<start>"]
        end_id = self.word_to_index["<end>"]

        dec_in = np.zeros((1, self.max_seq_length - 1), dtype="int64")
        dec_in[0, 0] = start_id

        neural_gloss = []
        for i in range(self.max_seq_length - 2):
            predictions = self.model(
                [enc_in, tf.convert_to_tensor(dec_in)], training=False
            ).numpy()
            predicted_id = int(np.argmax(predictions[0, i, :]))

            if predicted_id == end_id or predicted_id == 0:
                break

            predicted_word = self.index_to_word.get(predicted_id, "")
            if predicted_word and predicted_word != "[UNK]":
                neural_gloss.append(predicted_word)

            dec_in[0, i + 1] = predicted_id

        return " ".join(neural_gloss)

    def translate(self, english_sentence: str, mode: str = "hybrid") -> str:
        if not english_sentence or not english_sentence.strip():
            return ""

        if mode == "rule":
            gloss, _ = self.translate_rule_based(english_sentence)
            return gloss

        neural_output = self.translate_neural(english_sentence)
        if mode == "neural":
            return neural_output

        # Hybrid mode: Preserve neural output when it aligns with core semantics (>=60%),
        # while ensuring proper nouns / OOV tokens are retained in grammatical position.
        rule_output, spellable_tokens = self.translate_rule_based(english_sentence)
        rule_tokens = rule_output.split()
        neural_tokens = neural_output.split()

        expected_content = {w for w in rule_tokens if not w.startswith("X-")}
        neural_content = {w for w in neural_tokens if not w.startswith("X-")}

        if not expected_content:
            return neural_output if neural_output else rule_output

        overlap = len(expected_content.intersection(neural_content)) / len(expected_content)
        if overlap >= 0.60 and not spellable_tokens:
            return neural_output

        return rule_output