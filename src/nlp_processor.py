import spacy
from typing import List


class ASLTextProcessor:
    """
    Processes raw English text and converts it into a basic ASL Gloss sequence
    using linguistic rules and spaCy NLP pipeline.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        # Load the pretrained spaCy English language model
        self.nlp = spacy.load(model_name)

        # Define Part-of-Speech (POS) tags to ignore in ASL grammar
        # DET: Determiners (a, an, the)
        # AUX: Auxiliary verbs (is, am, are, was, were, have, has)
        # PUNCT: Punctuation marks (., !, ?)
        self.ignored_pos = {"DET", "AUX", "PUNCT"}

    def text_to_gloss(self, text: str) -> List[str]:
        """
        Converts an input English sentence into a list of ASL gloss tokens.
        """
        # Clean whitespace and pass the text through the spaCy NLP pipeline
        doc = self.nlp(text.strip())

        gloss_tokens = []
        time_tokens = []

        for token in doc:
            # Skip ignored grammatical categories and whitespace
            if token.pos_ in self.ignored_pos or token.is_space:
                continue

            # Extract the base dictionary form (lemma) and convert to uppercase
            lemma = token.lemma_.upper()

            # Normalize first-person pronouns to standard ASL gloss representation
            if lemma in {"I", "ME", "MY"}:
                lemma = "IX-1P"  # Index finger pointing to self (standard ASL notation)

            # In ASL grammar, time indicators usually move to the beginning of the sentence
            if token.ent_type_ in {"DATE", "TIME"}:
                time_tokens.append(lemma)
            else:
                gloss_tokens.append(lemma)

        # Combine time expressions first, followed by the main gloss sequence
        return time_tokens + gloss_tokens


# Quick test block when running this file directly
if __name__ == "__main__":
    processor = ASLTextProcessor()
    sample_sentences = [
        "Hello, how are you?",
        "I am going to the store tomorrow.",
        "She bought a new car yesterday.",
    ]

    for sentence in sample_sentences:
        gloss = processor.text_to_gloss(sentence)
        print(f"EN: {sentence}")
        print(f"ASL GLOSS: {' '.join(gloss)}\n")