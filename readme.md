# AI-Based Text-to-Sign Language Translator (ASL)

An end-to-end **Hybrid Neuro-Symbolic Deep Learning & NLP system** that translates written English sentences into **American Sign Language (ASL) Gloss** sequences and renders continuous subtitled sign language videos.

## System Architecture

```text
[English Text] ──► (Bidirectional LSTM + Luong Attention) ──► [Neural ASL Gloss]
                                                                      │
                   (spaCy Neuro-Symbolic Alignment & OOV Guard) ◄─────┘
                                    │
                                    ▼
                      [Verified ASL Gloss Sequence]
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
[Tier 1: MP4/GIF Signs]   [Tier 2: Spatial Pronouns]   [Tier 3: Fingerspelling]
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    ▼
          [OpenCV Subtitled Video Stitcher (outputs/translated_asl.mp4)]