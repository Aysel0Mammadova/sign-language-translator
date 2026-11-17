import os
from typing import List, Dict, Optional, Tuple


class ASLSignVisualizer:
    """
    Production-ready 3-Tier Hybrid ASL Visual Mapper.

    Architecture:
      - Tier 1: Word-level dynamic ASL signs (MP4 or GIF) -> data/signs/videos/
      - Tier 2: Word-level static ASL signs (PNG or JPG)  -> data/signs/words/
      - Grammatical Markers (e.g., IX-1P, IX-2P)          -> data/signs/grammatical/ (never fingerspelled)
      - Tier 3: Configurable Fingerspelling fallback      -> data/signs/alphabet/ (only if all letters exist)
    """

    def __init__(
        self,
        base_dir: str = os.path.join("data", "signs"),
        allow_fingerspelling: bool = True,
    ):
        self.base_dir = base_dir
        self.allow_fingerspelling = allow_fingerspelling

        # Dedicated asset directories
        self.videos_dir = os.path.join(base_dir, "videos")
        self.words_dir = os.path.join(base_dir, "words")
        self.grammatical_dir = os.path.join(base_dir, "grammatical")
        self.alphabet_dir = os.path.join(base_dir, "alphabet")

        # Ensure directory structure exists
        for directory in [
            self.videos_dir,
            self.words_dir,
            self.grammatical_dir,
            self.alphabet_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        # ASL grammatical indexing markers that represent spatial pointing, NOT spelled words
        # ASL grammatical indexing markers (both linguistic IX-* and ASLG-PC12 X-* notations)
        # Unified ASLG-PC12 spatial pronoun indexing markers (never fingerspelled)
        self.grammatical_markers = {
            "X-I": "Point to Self (I / Me)",
            "X-ME": "Point to Self (Me)",
            "X-MY": "Flat Hand on Chest (My / Mine)",
            "X-YOU": "Point to Listener (You)",
            "X-YOUR": "Flat Hand Toward Listener (Your)",
            "X-HE": "Point to Side (He)",
            "X-SHE": "Point to Side (She)",
            "X-IT": "Point to Side (It)",
            "X-WE": "Sweeping Point (We / Us)",
            "X-THEY": "Sweeping Point (They / Them)",
        }

    def _find_media_asset(
        self, directory: str, token: str, extensions: Tuple[str, ...]
    ) -> Optional[Tuple[str, str]]:
        """
        Searches for a matching media file in the given directory across allowed extensions.
        Returns (media_type, file_path) if found, otherwise None.
        """
        if not os.path.isdir(directory):
            return None

        for ext in extensions:
            candidate_path = os.path.join(directory, f"{token}.{ext}")
            if os.path.isfile(candidate_path):
                media_type = self._classify_extension(ext)
                return media_type, candidate_path
        return None

    @staticmethod
    def _classify_extension(ext: str) -> str:
        """Maps file extensions to standardized media types."""
        ext = ext.lower()
        if ext == "mp4":
            return "video"
        if ext == "gif":
            return "gif"
        if ext in {"png", "jpg", "jpeg"}:
            return "image"
        return "missing"

    def _handle_grammatical_marker(self, token: str) -> Dict[str, Optional[str]]:
        """
        Resolves ASL grammatical markers using dedicated video/GIF assets if available.
        Never falls back to fingerspelling.
        """
        description = self.grammatical_markers[token]
        asset = self._find_media_asset(
            self.grammatical_dir, token, ("mp4", "gif", "png", "jpg", "jpeg")
        )

        if asset:
            media_type, media_path = asset
            return {
                "gloss": token,
                "visual_type": "grammatical_marker",
                "media_type": media_type,
                "media_path": media_path,
                "label": f"{token} — {description}",
            }

        return {
            "gloss": token,
            "visual_type": "grammatical_marker",
            "media_type": "missing",
            "media_path": None,
            "label": f"{token} — {description} (Asset Missing)",
        }

    def _attempt_fingerspelling(self, token: str) -> Optional[List[Dict[str, Optional[str]]]]:
        """
        Attempts to spell out a token letter-by-letter using ASL alphabet images.
        Succeeds ONLY if every alphabetic character in the token has a valid image file.
        """
        letters = [char for char in token if char.isalpha()]
        if not letters:
            return None

        letter_assets = []
        for char in letters:
            asset = self._find_media_asset(self.alphabet_dir, char, ("png", "jpg", "jpeg"))
            if not asset:
                # Abort fingerspelling if any single letter image is missing
                return None
            media_type, media_path = asset
            letter_assets.append({
                "gloss": token,
                "visual_type": "fingerspelling",
                "media_type": media_type,
                "media_path": media_path,
                "label": f"{token} ({char})",
            })

        return letter_assets

    def gloss_to_visuals(
        self, gloss_sequence: str, allow_fingerspelling: Optional[bool] = None
    ) -> List[Dict[str, Optional[str]]]:
        """
        Converts an ASL Gloss sequence into an ordered list of visual sign items.

        Args:
            gloss_sequence: Space-separated uppercase ASL gloss string (e.g., 'TOMORROW IX-1P GO BAKU').
            allow_fingerspelling: Optional override for the instance's fallback policy.

        Returns:
            List of dictionaries preserving original Gloss token order.
        """
        if not gloss_sequence or not isinstance(gloss_sequence, str):
            return []

        use_fingerspelling = (
            self.allow_fingerspelling
            if allow_fingerspelling is None
            else allow_fingerspelling
        )

        tokens = gloss_sequence.strip().split()
        visual_sequence: List[Dict[str, Optional[str]]] = []

        for raw_token in tokens:
            token = raw_token.upper().strip()
            if not token:
                continue

            # 1. Handle Grammatical Markers separately (never fingerspell)
            if token in self.grammatical_markers:
                visual_sequence.append(self._handle_grammatical_marker(token))
                continue

            # 2. Tier 1: Word-level Dynamic ASL Videos (MP4 or GIF)
            tier1_asset = self._find_media_asset(self.videos_dir, token, ("mp4", "gif"))
            if tier1_asset:
                media_type, media_path = tier1_asset
                visual_sequence.append({
                    "gloss": token,
                    "visual_type": "tier1_dynamic_word",
                    "media_type": media_type,
                    "media_path": media_path,
                    "label": token,
                })
                continue

            # 3. Tier 2: Word-level Static ASL Images (PNG or JPG)
            tier2_asset = self._find_media_asset(self.words_dir, token, ("png", "jpg", "jpeg"))
            if tier2_asset:
                media_type, media_path = tier2_asset
                visual_sequence.append({
                    "gloss": token,
                    "visual_type": "tier2_static_word",
                    "media_type": media_type,
                    "media_path": media_path,
                    "label": token,
                })
                continue

            # 4. Tier 3: Configurable Fingerspelling Fallback (Atomic check)
            if use_fingerspelling:
                spelled_items = self._attempt_fingerspelling(token)
                if spelled_items:
                    visual_sequence.extend(spelled_items)
                    continue

            # 5. Explicitly report missing ASL sign when no valid visual asset exists
            visual_sequence.append({
                "gloss": token,
                "visual_type": "unmapped_word",
                "media_type": "missing",
                "media_path": None,
                "label": f"{token} (Sign Not Found)",
            })

        return visual_sequence


if __name__ == "__main__":
    visualizer = ASLSignVisualizer(allow_fingerspelling=True)
    sample_gloss = "TOMORROW IX-1P GO BAKU"

    print("=== Policy A: Fingerspelling ENABLED ===")
    for item in visualizer.gloss_to_visuals(sample_gloss, allow_fingerspelling=True):
        print(item)

    print("\n=== Policy B: Fingerspelling DISABLED (Strict Dictionary Mode) ===")
    for item in visualizer.gloss_to_visuals(sample_gloss, allow_fingerspelling=False):
        print(item)