import os
import string
from PIL import Image
from datasets import load_dataset

# Note: In ASL, letters 'J' and 'Z' involve dynamic wrist/finger motion.
# Static images represent their initial/final handshape posture.
DYNAMIC_ALPHABET_LETTERS = {"J", "Z"}


def setup_asl_alphabet_images():
    """
    Streams authentic ASL Alphabet hand-sign images from Hugging Face
    and saves one clean reference image per letter (A-Z) into 'data/signs/alphabet/'.
    Raises RuntimeError if authentic images cannot be retrieved (never generates fake cards).
    """
    alphabet_dir = os.path.join("data", "signs", "alphabet")
    os.makedirs(alphabet_dir, exist_ok=True)

    print("Streaming authentic ASL Alphabet dataset from Hugging Face...")
    dataset = load_dataset(
        "Marxulia/asl_sign_languages_alphabets_v03",
        split="train",
        streaming=True
    )

    # Safely resolve label mapping even when streaming features are lazily initialized
    label_names = None
    if getattr(dataset, "features", None) and "label" in dataset.features:
        label_names = dataset.features["label"].names

    saved_letters = set()
    for sample in dataset:
        raw_label = sample.get("label")
        if label_names is not None and isinstance(raw_label, int):
            letter = label_names[raw_label].upper()
        else:
            letter = str(raw_label).upper()

        if letter in string.ascii_uppercase and letter not in saved_letters:
            img: Image.Image = sample["image"].convert("RGB").resize((256, 256))
            save_path = os.path.join(alphabet_dir, f"{letter}.png")
            img.save(save_path)
            saved_letters.add(letter)
            motion_note = " (Dynamic motion letter - keyframe)" if letter in DYNAMIC_ALPHABET_LETTERS else ""
            print(f"  [SAVED] {letter}.png{motion_note}")

        if len(saved_letters) == 26:
            break

    if len(saved_letters) < 26:
        missing = sorted(set(string.ascii_uppercase) - saved_letters)
        raise RuntimeError(
            f"Incomplete ASL alphabet dataset. Missing letters: {missing}. "
            "Check network connection or provide authentic ASL hand-sign images manually."
        )

    print(f"\nSuccess! All 26 authentic ASL hand-sign images are verified in '{alphabet_dir}'.")


if __name__ == "__main__":
    setup_asl_alphabet_images()