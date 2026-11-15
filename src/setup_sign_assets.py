import os
import string
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset


def setup_asl_alphabet_images():
    """
    Downloads real ASL Alphabet hand-sign images from Hugging Face
    and saves one clean reference image per letter (A-Z) into 'data/signs/alphabet/'.
    """
    alphabet_dir = os.path.join("data", "signs", "alphabet")
    words_dir = os.path.join("data", "signs", "words")
    os.makedirs(alphabet_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)

    print("Fetching real ASL Alphabet dataset from Hugging Face...")
    try:
        # Load a lightweight public ASL alphabet image dataset
        dataset = load_dataset("Marxulia/asl_sign_languages_alphabets_v03", split="train", streaming=True)
        label_names = dataset.features["label"].names

        saved_letters = set()
        for sample in dataset:
            label_idx = sample["label"]
            letter = label_names[label_idx].upper()

            # Save only standard A-Z letters (one image per letter)
            if letter in string.ascii_uppercase and letter not in saved_letters:
                img: Image.Image = sample["image"].convert("RGB")
                img = img.resize((256, 256))
                save_path = os.path.join(alphabet_dir, f"{letter}.png")
                img.save(save_path)
                saved_letters.add(letter)
                print(f"  Saved real ASL hand-sign image for '{letter}' -> {save_path}")

            if len(saved_letters) == 26:
                break

        print(f"\nSuccessfully saved {len(saved_letters)} real ASL hand-sign images!")

    except Exception as e:
        print(f"Dataset stream interrupted ({e}). Generating fallback visual cards...")
        for letter in string.ascii_uppercase:
            save_path = os.path.join(alphabet_dir, f"{letter}.png")
            if not os.path.exists(save_path):
                img = Image.new("RGB", (256, 256), color=(240, 244, 248))
                draw = ImageDraw.Draw(img)
                draw.rectangle([10, 10, 246, 246], outline=(30, 64, 175), width=4)
                draw.text((105, 100), letter, fill=(15, 23, 42))
                draw.text((70, 160), f"ASL SIGN: {letter}", fill=(71, 85, 105))
                img.save(save_path)


if __name__ == "__main__":
    setup_asl_alphabet_images()