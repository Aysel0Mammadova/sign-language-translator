import os
from typing import List, Dict, Optional, Tuple


class ASLSignVisualizer:
    """
    Unified 3-Tier Hybrid ASL Visual Mapper (Standardized X-* Notation):
      - Tier 1: Word-level dynamic ASL videos (MP4/GIF) -> data/signs/videos/
      - Tier 2: Word-level static ASL images (PNG/JPG)  -> data/signs/words/
      - Grammatical Markers (X-I, X-YOU, etc.)          -> data/signs/grammatical/ (with visual instruction fallback)
      - Tier 3: Fingerspelling fallback                 -> data/signs/alphabet/ (only single-word tokens, never hyphenated compounds)
    """

    def __init__(
        self,
        base_dir: str = os.path.join("data", "signs"),
        allow_fingerspelling: bool = True,
    ):
        self.videos_dir = os.path.join(base_dir, "videos")
        self.words_dir = os.path.join(base_dir, "words")
        self.grammatical_dir = os.path.join(base_dir, "grammatical")
        self.alphabet_dir = os.path.join(base_dir, "alphabet")
        self.allow_fingerspelling = allow_fingerspelling

        for d in [self.videos_dir, self.words_dir, self.grammatical_dir, self.alphabet_dir]:
            os.makedirs(d, exist_ok=True)

        self.grammatical_markers = {
            "X-I": "Point Index Finger to Chest (I / Me)",
            "X-ME": "Point Index Finger to Chest (Me)",
            "X-MY": "Place Flat Hand on Chest (My / Mine)",
            "X-YOU": "Point Index Finger Toward Listener (You)",
            "X-YOUR": "Open Palm Toward Listener (Your)",
            "X-HE": "Point Index Finger to Side (He)",
            "X-SHE": "Point Index Finger to Side (She)",
            "X-IT": "Point Index Finger to Side (It)",
            "X-WE": "Arc Point Across Chest (We / Us)",
            "X-THEY": "Sweeping Point to Side (They / Them)",
        }

    def _find_media(self, directory: str, token: str, exts: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
        for ext in exts:
            path = os.path.join(directory, f"{token}.{ext}")
            if os.path.isfile(path):
                media_type = "video" if ext == "mp4" else ("gif" if ext == "gif" else "image")
                return media_type, path
        return None

    def _attempt_fingerspelling(self, token: str) -> Optional[List[Dict[str, Optional[str]]]]:
        # Never fingerspell hyphenated compound glosses (e.g., THANK-YOU) or grammatical prefixes
        if "-" in token or not token.isalpha():
            return None

        items = []
        for char in token:
            asset = self._find_media(self.alphabet_dir, char, ("png", "jpg", "jpeg"))
            if not asset:
                return None
            media_type, path = asset
            items.append({
                "gloss": token,
                "visual_type": "fingerspelling",
                "media_type": media_type,
                "media_path": path,
                "label": f"{token} ({char})",
            })
        return items

    def gloss_to_visuals(
        self, gloss_sequence: str, allow_fingerspelling: Optional[bool] = None
    ) -> List[Dict[str, Optional[str]]]:
        if not gloss_sequence:
            return []

        use_spelling = self.allow_fingerspelling if allow_fingerspelling is None else allow_fingerspelling
        visual_sequence = []

        for raw_tok in gloss_sequence.strip().split():
            token = raw_tok.upper().strip()
            if not token:
                continue

            # 1. Grammatical Spatial Markers
            if token in self.grammatical_markers:
                desc = self.grammatical_markers[token]
                asset = self._find_media(self.grammatical_dir, token, ("mp4", "gif", "png"))
                visual_sequence.append({
                    "gloss": token,
                    "visual_type": "grammatical_marker",
                    "media_type": asset[0] if asset else "instruction_card",
                    "media_path": asset[1] if asset else None,
                    "label": f"{token}: {desc}",
                })
                continue

            # 2. Tier 1: Dynamic Word Videos
            t1 = self._find_media(self.videos_dir, token, ("mp4", "gif"))
            if t1:
                visual_sequence.append({
                    "gloss": token,
                    "visual_type": "tier1_dynamic_word",
                    "media_type": t1[0],
                    "media_path": t1[1],
                    "label": token,
                })
                continue

            # 3. Tier 2: Static Word Images
            t2 = self._find_media(self.words_dir, token, ("png", "jpg", "jpeg"))
            if t2:
                visual_sequence.append({
                    "gloss": token,
                    "visual_type": "tier2_static_word",
                    "media_type": t2[0],
                    "media_path": t2[1],
                    "label": token,
                })
                continue

            # 4. Tier 3: Fingerspelling Fallback (Single-word tokens only)
            if use_spelling:
                spelled = self._attempt_fingerspelling(token)
                if spelled:
                    visual_sequence.extend(spelled)
                    continue

            # 5. Unmapped Token
            visual_sequence.append({
                "gloss": token,
                "visual_type": "unmapped_word",
                "media_type": "missing",
                "media_path": None,
                "label": f"{token} (Sign Not Found)",
            })

        return visual_sequence


if __name__ == "__main__":
    vis = ASLSignVisualizer()
    for item in vis.gloss_to_visuals("TOMORROW X-I TRAVEL BAKU"):
        print(item)