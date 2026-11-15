import os
import sys
import streamlit as st

# Add 'src' directory to Python path so we can import our modular classes cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer


# 1. Page Configuration
st.set_page_config(
    page_title="AI Sign Language Translator (ASL)",
    page_icon="🤟",
    layout="wide"
)


# 2. Cache the Deep Learning Model (loads once into memory for fast inference)
@st.cache_resource
def load_translation_engine():
    return ASLTranslator(model_dir="models")


def render_visual_card(col, item: dict):
    """
    Renders a single ASL visual sign card inside a Streamlit column
    based on its media_type (video, gif, image, or missing).
    """
    with col:
        media_type = item["media_type"]
        media_path = item["media_path"]
        label = item["label"]
        visual_type = item["visual_type"]

        # Visual badge indicating the tier/source of the sign
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
            st.image(media_path, use_container_width=True)
            st.markdown(f"**{label}**")

        else:
            # Clean placeholder card for grammatical pointers or missing dictionary words
            st.info(f"**{label}**")


def main():
    # --- Sidebar Settings ---
    st.sidebar.title("⚙️ System Settings")
    allow_fingerspelling = st.sidebar.toggle(
        "Enable Fingerspelling Fallback",
        value=True,
        help="When enabled, unknown words and proper nouns are spelled letter-by-letter if all alphabet images exist."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Architecture Overview")
    st.sidebar.markdown(
        """
        - **NLP & Translation:** TensorFlow Seq2Seq + Luong Attention
        - **Dataset:** ASLG-PC12 Parallel Corpus
        - **Visual Engine:** 3-Tier Hybrid Mapper (MP4/GIF → Word PNG → Fingerspelling)
        """
    )

    # --- Main Header ---
    st.title("🤟 AI-Based Text to Sign Language Translator (ASL)")
    st.markdown(
        "Enter an English sentence below. The Deep Learning model translates it into **ASL Gloss** "
        "(respecting ASL grammar), and the visual engine maps each token to sign representations."
    )

    # Load backend engines
    try:
        translator = load_translation_engine()
    except Exception as e:
        st.error(f"Failed to load the trained model from 'models/'. Error: {e}")
        st.stop()

    visualizer = ASLSignVisualizer(allow_fingerspelling=allow_fingerspelling)

    # --- User Input Section ---
    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Quick Test Sentences:**")
        example_choice = st.selectbox(
            "Choose a sample:",
            [
                "Custom Input...",
                "thank you very much for your help",
                "what is the problem today",
                "we must vote tomorrow",
                "approval of minutes of previous sitting",
            ],
            label_visibility="collapsed"
        )

    with col_input:
        default_text = "" if example_choice == "Custom Input..." else example_choice
        user_text = st.text_input(
            "Enter English Sentence:",
            value=default_text,
            placeholder="e.g., I am going to school tomorrow..."
        )

    translate_btn = st.button("Translate to ASL", type="primary")

    # --- Translation & Visualization Pipeline ---
    if translate_btn and user_text.strip():
        with st.spinner("Translating English to ASL Gloss..."):
            gloss_output = translator.translate(user_text)

        if not gloss_output:
            st.warning("Could not generate a valid ASL Gloss sequence for this input.")
            return

        # Display Intermediate Linguistic Representation (ASL Gloss)
        st.subheader("1. Intermediate Representation (ASL Gloss)")
        st.code(gloss_output, language="text")

        # Map Gloss to Visual Sign Assets
        visual_items = visualizer.gloss_to_visuals(gloss_output)

        st.subheader("2. Visual Sign Language Sequence")
        if not visual_items:
            st.warning("No visual signs could be mapped.")
            return

        # Display visual cards in clean rows of 5 columns
        cols_per_row = 5
        for i in range(0, len(visual_items), cols_per_row):
            row_items = visual_items[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, item in zip(cols, row_items):
                render_visual_card(col, item)

    elif translate_btn:
        st.warning("Please enter an English sentence first.")


if __name__ == "__main__":
    main()