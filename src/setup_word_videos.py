import os
import re
import time
import requests

# Core everyday & academic ASL vocabulary words to fetch real sign videos for
TARGET_WORDS = [
    "HELLO", "STUDENT", "DOCTOR", "TEACHER", "SCHOOL", "UNIVERSITY",
    "BOOK", "HELP", "LOVE", "YES", "NO", "WHAT", "WHERE", "NAME",
    "LEARN", "STUDY", "WORK", "FRIEND", "TOMORROW", "YESTERDAY",
    "THANK-YOU", "WELCOME", "UNDERSTAND", "LIVE", "GO", "BUY", "CAR",
    "NEW", "NEED", "PLEASE", "GOOD", "MORNING", "EVENING", "READ",
    "HOSPITAL", "TRAVEL", "SIGN", "LANGUAGE", "PROBLEM", "TODAY", "VOTE", "WILL"
]


def fetch_signasl_video(word: str, save_dir: str) -> bool:
    """
    Fetches an authentic ASL sign video (.mp4) for a given English/Gloss word
    from the public SignASL dictionary and saves it locally.
    """
    query_word = word.lower().replace("-", " ")
    page_url = f"https://www.signasl.org/sign/{query_word.replace(' ', '-')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(page_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return False

        # Extract direct .mp4 video URL from the page HTML
        mp4_matches = re.findall(r'https://[^\s"\'<>]+\.mp4', resp.text)
        if not mp4_matches:
            return False

        video_url = mp4_matches[0]
        video_resp = requests.get(video_url, headers=headers, timeout=15)
        if video_resp.status_code == 200 and len(video_resp.content) > 5000:
            save_path = os.path.join(save_dir, f"{word.upper()}.mp4")
            with open(save_path, "wb") as f:
                f.write(video_resp.content)
            return True
    except Exception:
        return False

    return False


def download_asl_videos():
    videos_dir = os.path.join("data", "signs", "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print(f"Downloading real ASL word-level videos into '{videos_dir}'...\n")
    downloaded = 0

    for word in TARGET_WORDS:
        save_path = os.path.join(videos_dir, f"{word}.mp4")
        if os.path.exists(save_path):
            print(f"  [EXISTS] {word}.mp4")
            downloaded += 1
            continue

        success = fetch_signasl_video(word, videos_dir)
        if success:
            print(f"  [SAVED]  {word}.mp4 (Real ASL Video)")
            downloaded += 1
        else:
            print(f"  [SKIP]   No direct video found for '{word}'")

        time.sleep(0.4)  # Polite delay between requests

    print(f"\nDone! {downloaded}/{len(TARGET_WORDS)} real ASL word videos are ready in '{videos_dir}'.")


if __name__ == "__main__":
    download_asl_videos()