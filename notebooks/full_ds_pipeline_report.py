import os
import sys
import cv2
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from evaluate import evaluate_benchmark, UNSEEN_CONVERSATIONAL_BENCHMARK
from translator import ASLTranslator
from sign_visualizer import ASLSignVisualizer


def audit_multimodal_data_formats():
    """
    Covers Requirement #2 (Multimodal EDA) & Requirement #3 (Format, Resolution & FPS Validation)
    across Text (CSV), Image (PNG), and Video (MP4) modalities.
    """
    print("=" * 70)
    print("REQUIREMENT 1-4: DATA LOADING, FORMAT VALIDATION & CLEANING AUDIT")
    print("=" * 70)

    # 1. Tabular / Parallel Text Corpus Audit
    csv_path = os.path.join("data", "processed_asl.csv")
    df = pd.read_csv(csv_path)
    null_count = df.isnull().sum().sum()
    dup_count = df.duplicated(subset=["english"]).sum()
    print(f"[TEXT CORPUS] '{csv_path}': {len(df)} rows | Nulls: {null_count} | Duplicates: {dup_count}")

    # 2. Image Data Audit (Alphabet Hand-Signs)
    alpha_dir = os.path.join("data", "signs", "alphabet")
    img_files = [f for f in os.listdir(alpha_dir) if f.endswith(".png")]
    resolutions = set()
    corrupt_images = 0
    for f in img_files:
        p = os.path.join(alpha_dir, f)
        if os.path.getsize(p) == 0:
            corrupt_images += 1
            continue
        with Image.open(p) as im:
            resolutions.add((im.size, im.mode))
    print(f"[IMAGE ASSETS] '{alpha_dir}': {len(img_files)} PNG files | Resolutions/Modes: {resolutions} | Corrupt: {corrupt_images}")

    # 3. Video Data Audit (Dynamic Word-Level ASL Videos)
    vid_dir = os.path.join("data", "signs", "videos")
    vid_files = [f for f in os.listdir(vid_dir) if f.endswith(".mp4")]
    fps_list = []
    vid_shapes = set()
    corrupt_videos = 0

    for f in vid_files:
        p = os.path.join(vid_dir, f)
        if os.path.getsize(p) < 1000:
            corrupt_videos += 1
            continue
        cap = cv2.VideoCapture(p)
        fps = round(cap.get(cv2.CAP_PROP_FPS), 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_cnt == 0:
            corrupt_videos += 1
        else:
            fps_list.append(fps)
            vid_shapes.add((w, h))

    fps_range = f"{min(fps_list)}-{max(fps_list)}" if fps_list else "N/A"
    print(f"[VIDEO ASSETS] '{vid_dir}': {len(vid_files)} MP4 files | Unique Resolutions: {vid_shapes} | FPS Range: {fps_range} | Corrupt: {corrupt_videos}")
    print("-> Note: Requirement #5 (Preprocessing) unifies all heterogeneous video resolutions/FPS into 480x368 @ 24 FPS H.264.\n")


if __name__ == "__main__":
    audit_multimodal_data_formats()

    print("=" * 70)
    print("REQUIREMENT 7-10: SPLIT VERIFICATION, MULTI-MODEL TEST & EVALUATION")
    print("=" * 70)
    translator = ASLTranslator()
    visualizer = ASLSignVisualizer()

    test_df = pd.read_csv(os.path.join("data", "test_split.csv")).dropna()
    aslg_pairs = [
        (r["english"], r["gloss"].replace("<start>", "").replace("<end>", "").strip())
        for _, r in test_df.iterrows()
    ]
    evaluate_benchmark(aslg_pairs, translator, visualizer, "Full ASLG-PC12 Test Split")
    evaluate_benchmark(UNSEEN_CONVERSATIONAL_BENCHMARK, translator, visualizer, "100% Unseen Conversational Benchmark")