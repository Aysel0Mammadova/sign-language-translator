import os
import cv2
import numpy as np
import imageio.v2 as imageio
from typing import List, Dict


def _letterbox_frame(frame: np.ndarray, target_size: tuple = (480, 360)) -> np.ndarray:
    """
    Resizes an image/frame while preserving aspect ratio by adding neutral padding bars
    (letterboxing) so hand shapes are never stretched or distorted.
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), (30, 41, 59), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def _create_instruction_card(label: str, target_size: tuple = (480, 360)) -> np.ndarray:
    """Renders a clean visual instruction frame for grammatical markers (e.g., X-I, X-YOU)."""
    w, h = target_size
    card = np.full((h, w, 3), (30, 58, 138), dtype=np.uint8)
    cv2.rectangle(card, (20, 20), (w - 20, h - 20), (147, 197, 253), 2)

    parts = label.split(": ")
    title = parts[0]
    desc = parts[1] if len(parts) > 1 else ""

    cv2.putText(card, title, (45, h // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(card, desc, (45, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (219, 234, 254), 1, cv2.LINE_AA)
    return card


def _draw_semitransparent_subtitle(frame: np.ndarray, text: str) -> np.ndarray:
    """Blends a true semi-transparent dark overlay bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 42), (w, h), (15, 23, 42), -1)
    blended = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    cv2.putText(
        blended,
        text,
        (16, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return blended


def render_sequence_video(
    visual_items: List[Dict],
    output_path: str = os.path.join("outputs", "translated_asl.mp4"),
    frame_size: tuple = (480, 368),
    fps: int = 24,
    card_duration_sec: float = 1.2,
) -> str:
    """
    Streams dynamic videos, GIFs, static images, and grammatical instruction cards
    directly into a browser-compatible H.264 MP4 file without buffering all frames in RAM.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_written = 0

    with imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for item in visual_items:
            media_type = item.get("media_type")
            media_path = item.get("media_path")
            label = item.get("label", "")

            if media_type == "instruction_card":
                card_bgr = _create_instruction_card(label, frame_size)
                card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
                for _ in range(int(fps * card_duration_sec)):
                    writer.append_data(card_rgb)
                    frames_written += 1

            elif media_type in {"video", "gif"} and media_path and os.path.exists(media_path):
                try:
                    reader = imageio.get_reader(media_path)
                    for frame_rgb in reader:
                        if frame_rgb is None or frame_rgb.size == 0:
                            continue
                        frame_bgr = cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
                        boxed = _letterbox_frame(frame_bgr, frame_size)
                        subtitled = _draw_semitransparent_subtitle(boxed, label)
                        writer.append_data(cv2.cvtColor(subtitled, cv2.COLOR_BGR2RGB))
                        frames_written += 1
                    reader.close()
                except Exception as e:
                    print(f"[WARNING] Failed to read media '{media_path}': {e}")

            elif media_type == "image" and media_path and os.path.exists(media_path):
                img_bgr = cv2.imread(media_path)
                if img_bgr is not None and img_bgr.size > 0:
                    boxed = _letterbox_frame(img_bgr, frame_size)
                    subtitled = _draw_semitransparent_subtitle(boxed, label)
                    img_rgb = cv2.cvtColor(subtitled, cv2.COLOR_BGR2RGB)
                    for _ in range(int(fps * card_duration_sec)):
                        writer.append_data(img_rgb)
                        frames_written += 1

    if frames_written == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return ""

    return output_path