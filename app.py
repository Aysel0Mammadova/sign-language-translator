import os
import sys
import tempfile
import cv2
import streamlit as st

# Add 'src' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer


# 1. Page Configuration
st.set_page_config(
    page_title="AI Sign Language Translator (ASL)",
    page_icon="🤟",
    layout="wide"
)


# 2. Cache the Deep Learning Model
@st.cache_resource
def load_translation_engine():
    return ASLTranslator(model_dir="models")


def stitch_sign_videos(visual_items: list) -> str:
    """
    Concatenates all available MP4 sign videos in the translated sequence
    into a single continuous sentence video using OpenCV.
    """
    video_paths = [
        item["media_path"]
        for item in visual_items
        if item["media_type"] == "video" and item["media_path"] and os.path.exists(item["media_path"])
    ]

    if len(video_paths) < 2:
        return ""

    target_size = (320, 240)
    fps = 24.0
    temp_out = os.path.join(tempfile.gettempdir(), "asl_combined_sentence.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, target_size)

    # Fallback codec if avc1 is unavailable on the host OS
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_out, fourcc, fps, target_size)

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, target_size)
            writer.write(resized_frame)
        cap.release()

    writer.release()
    return temp_out if os.path.exists(temp_out) else ""


def render_visual_card(col, item: dict):
    """Renders an individual ASL sign card inside a grid column."""
    with col:
        media_type = item["media_type"]
        media_path = item["media_path"]
        label = item["label"]
        visual_type = item["visual_type"]

        badge_map = {
            "tier1_dynamic_word": "🎥 Dynamic Sign",
            "tier2_static_word": "🖼️ Word Sign",
            "grammatical_marker": "👉 Grammar Marker",
            "fingerspelling": "🔤 Fingerspelling",
            "unmapped_word": "⚠️ Unmapped",
        }
        st.caption(badge_map.get(visual_type, "Sign"))

        if media_type == "video" and media_path:
            st.video(media_path, autoplay=True, loop=True)
            st.markdown(f"**{label}**")
        elif media_type in {"gif", "image"} and media_path:
            st.image(media_path, width="stretch")
            st.markdown(f"**{label}**")
        else:
            st.info(f"**{label}**")


def main():
    # --- Sidebar Settings ---
    st.sidebar.title("⚙️ System Configuration")
    allow_fingerspelling = st.sidebar.toggle(
        "Enable Fingerspelling Fallback",
        value=True,
        help="Spell out-of-vocabulary words letter-by-letter if all alphabet assets exist."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model & System Specs")
    st.sidebar.markdown(
        """
        - **Architecture:** Bidirectional LSTM + Luong Attention
        - **Guardrails:** Neuro-Symbolic `spaCy` Alignment
        - **Corpus:** ASLG-PC12 + Conversational Domain
        - **Visual Engine:** 3-Tier Hybrid Mapper + OpenCV Stitcher
        """
    )

    st.title("🤟 AI-Based Text to Sign Language Translator (ASL)")

    try:
        translator = load_translation_engine()
    except Exception as e:
        st.error(f"Failed to load trained model from 'models/': {e}")
        st.stop()

    visualizer = ASLSignVisualizer(allow_fingerspelling=allow_fingerspelling)

    # --- Navigation Tabs ---
    tab_translate, tab_analytics, tab_dictionary = st.tabs([
        "🌐 Live Translator",
        "📊 Dataset & Model Analytics",
        "📚 Sign Video Dictionary"
    ])

    # =========================================================================
    # TAB 1: LIVE TRANSLATOR
    # =========================================================================
    with tab_translate:
        st.markdown(
            "Translate English sentences into **ASL Gloss** and visualize them through "
            "dynamic sign videos, grammatical markers, and fingerspelling."
        )

        col_input, col_examples = st.columns([3, 1])
        with col_examples:
            st.markdown("**Sample Sentences:**")
            example_choice = st.selectbox(
                "Sample:",
                [
                    "Custom Input...",
                    "I am a doctor",
                    "I will travel tomorrow",
                    "Where do you work",
                    "Thank you very much",
                    "I love learning sign language",
                ],
                label_visibility="collapsed"
            )

        with col_input:
            default_text = "" if example_choice == "Custom Input..." else example_choice
            user_text = st.text_input(
                "Enter English Sentence:",
                value=default_text,
                placeholder="Type an English sentence here..."
            )

        if st.button("Translate to ASL", type="primary"):
            if not user_text.strip():
                st.warning("Please enter an English sentence first.")
            else:
                with st.spinner("Translating and mapping visual signs..."):
                    gloss_output = translator.translate(user_text)
                    visual_items = visualizer.gloss_to_visuals(gloss_output)

                st.subheader("1. Intermediate Representation (ASL Gloss)")
                st.code(gloss_output, language="text")

                # Combined Sentence Video Player (when 2+ dynamic videos exist)
                combined_video_path = stitch_sign_videos(visual_items)
                if combined_video_path:
                    st.subheader("2. Continuous Sign Sequence Player")
                    col_vid, _ = st.columns([1, 2])
                    with col_vid:
                        st.video(combined_video_path, autoplay=True, loop=True)

                st.subheader("3. Token-by-Token Visual Breakdown")
                cols_per_row = 5
                for i in range(0, len(visual_items), cols_per_row):
                    row_items = visual_items[i : i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, item in zip(cols, row_items):
                        render_visual_card(col, item)

    # =========================================================================
    # TAB 2: DATASET & MODEL ANALYTICS
    # =========================================================================
    with tab_analytics:
        st.subheader("Exploratory Data Analysis & Evaluation Metrics")
        st.markdown(
            "Quantitative evaluation on unseen test splits demonstrates how ASL Gloss condenses "
            "English syntax by dropping articles and auxiliary verbs."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Training Corpus Size", "35,000+ Pairs", "ASLG-PC12 + Domain")
        m2.metric("Validation Accuracy", "~80.0%", "Bi-LSTM + Attention")
        m3.metric("Architecture Type", "Neuro-Symbolic", "TF + spaCy Guardrails")

        eda_chart_path = os.path.join("assets", "dataset_eda.png")
        if os.path.exists(eda_chart_path):
            st.image(eda_chart_path, caption="ASLG-PC12 Token Distribution & Length Comparison", width="stretch")
        else:
            st.info("Run 'python notebooks/generate_report_charts.py' to generate EDA charts.")

    # =========================================================================
    # TAB 3: SIGN VIDEO DICTIONARY
    # =========================================================================
    with tab_dictionary:
        st.subheader("Available Dynamic ASL Word Videos")
        videos_dir = os.path.join("data", "signs", "videos")
        if os.path.isdir(videos_dir):
            available_videos = sorted([f for f in os.listdir(videos_dir) if f.endswith((".mp4", ".gif"))])
            st.write(f"Total dynamic sign assets loaded: **{len(available_videos)}**")
            st.write(", ".join([f"`{v.split('.')[0]}`" for v in available_videos]))
        else:
            st.warning("No dynamic video directory found.")


if __name__ == "__main__":
    main()