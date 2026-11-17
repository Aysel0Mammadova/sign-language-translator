import os
import cv2
from typing import List, Dict


def render_sequence_video(
    visual_items: List[Dict],
    output_path: str = os.path.join("outputs", "translated_asl.mp4"),
    frame_size: tuple = (480, 360),
    fps: float = 24.0,
    static_duration_sec: float = 1.0,
) -> str:
    """
    Combines dynamic sign videos (MP4/GIF) and static sign images (PNG/JPG)
    in sequence order into a single continuous MP4 video with gloss subtitle overlays.

    Returns the output file path if at least one visual frame was written, else ''.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frames_written = 0
    for item in visual_items:
        media_type = item.get("media_type")
        media_path = item.get("media_path")
        label = item.get("label", "")

        if not media_path or not os.path.exists(media_path):
            continue

        if media_type in {"video", "gif"}:
            cap = cv2.VideoCapture(media_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, frame_size)
                _draw_subtitle(resized, label)
                writer.write(resized)
                frames_written += 1
            cap.release()

        elif media_type == "image":
            img = cv2.imread(media_path)
            if img is not None:
                resized = cv2.resize(img, frame_size)
                _draw_subtitle(resized, label)
                for _ in range(int(fps * static_duration_sec)):
                    writer.write(resized)
                    frames_written += 1

    writer.release()
    return output_path if frames_written > 0 else ""


def _draw_subtitle(frame, text: str):
    """Draws a clean semi-transparent subtitle banner at the bottom of a video frame."""
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 40), (w, h), (15, 23, 42), -1)
    cv2.putText(
        frame,
        text,
        (15, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )