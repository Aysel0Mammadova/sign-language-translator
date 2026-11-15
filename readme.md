#  AI-Based Text-to-Sign Language Translator (ASL)

An end-to-end **Hybrid Neuro-Symbolic Deep Learning & NLP system** that translates written English sentences into **American Sign Language (ASL) Gloss** sequences and renders visual sign language representations through dynamic videos, grammatical markers, and fingerspelling.

## System Architecture

```text
[English Text] ──► (Bidirectional LSTM + Luong Attention) ──► [Raw ASL Gloss]
                                                                    │
                   (spaCy Neuro-Symbolic Guardrails) ◄──────────────┘
                                  │
                                  ▼
                    [Verified ASL Gloss Sequence]
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
[Tier 1: MP4/GIF]      [Tier 2: Grammar Markers]   [Tier 3: Fingerspelling]
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
             [OpenCV Continuous Video Stitcher & Streamlit UI]