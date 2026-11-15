# AI-Based Text-to-Sign Language Translator (ASL)

An end-to-end **Hybrid Neuro-Symbolic Deep Learning & NLP system** that translates written English sentences into **American Sign Language (ASL) Gloss** sequences and renders corresponding visual sign language representations.

## System Architecture

1. **Data Pipeline (`src/prepare_data.py`):** Processes the **ASLG-PC12** parallel corpus merged with conversational domain pairs, applying Train/Validation/Test splits without data leakage.
2. **Translation Engine (`src/train_pipeline.py` & `src/translator.py`):**
   - **Bidirectional Encoder-Decoder LSTM + Luong Attention** implemented in TensorFlow/Keras.
   - **Neuro-Symbolic Guardrails (`spaCy`):** Enforces ASL grammar rules (Time-Topic-Comment ordering, pronoun indexing like `X-I`, `X-YOU`, and OOV hallucination protection).
3. **Evaluation (`src/evaluate.py`):** Computes corpus-level **BLEU-1** and **BLEU-4** scores on unseen test splits.
4. **3-Tier Hybrid Visual Mapper (`src/sign_visualizer.py`):**
   - **Tier 1:** Dynamic word-level ASL sign videos (`.mp4` / `.gif`).
   - **Tier 2:** Static word-level sign illustrations (`.png` / `.jpg`).
   - **Tier 3:** Configurable fingerspelling fallback with atomic letter validation.
5. **Interactive Web Application (`app.py`):** Built with **Streamlit** for real-time translation and configurable fallback modes.


```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Prepare data & train model
python src/prepare_data.py
python src/train_pipeline.py

# 4. Download ASL visual assets
python src/setup_sign_assets.py
python src/setup_word_videos.py

# 5. Launch the web application
streamlit run app.py