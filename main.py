import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer
from video_stitcher import render_sequence_video


def run_end_to_end_pipeline(
    english_text: str,
    allow_fingerspelling: bool = True,
    output_video: str = os.path.join("outputs", "translated_asl.mp4"),
) -> dict:
    """
    Executes the complete Text -> ASL Gloss -> Visual Mapping -> Stitched Video pipeline.
    """
    translator = ASLTranslator(model_dir="models")
    visualizer = ASLSignVisualizer(allow_fingerspelling=allow_fingerspelling)

    gloss_sequence = translator.translate(english_text, mode="hybrid")
    visual_items = visualizer.gloss_to_visuals(gloss_sequence)
    video_path = render_sequence_video(visual_items, output_path=output_video)

    return {
        "input_english": english_text,
        "asl_gloss": gloss_sequence,
        "visual_sequence": visual_items,
        "output_video": video_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End English to ASL Translator Pipeline")
    parser.add_argument(
        "--text",
        type=str,
        default="I will travel tomorrow",
        help="English sentence to translate into ASL",
    )
    parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback for unmapped words",
    )
    args = parser.parse_args()

    result = run_end_to_end_pipeline(
        english_text=args.text,
        allow_fingerspelling=not args.no_fingerspelling,
    )

    print("\n" + "=" * 50)
    print(f"Input English : {result['input_english']}")
    print(f"ASL Gloss     : {result['asl_gloss']}")
    print(f"Mapped Items  : {len(result['visual_sequence'])}")
    print(f"Stitched Video: {result['output_video'] or 'No visual media available'}")
    print("=" * 50)