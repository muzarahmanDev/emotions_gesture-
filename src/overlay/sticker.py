from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence


@dataclass
class StickerAsset:
    frames: list[np.ndarray]       # Daftar frame BGRA
    frame_durations_ms: list[int]  # Durasi tiap frame (ms)

    def get_frame(self, elapsed_seconds: float) -> np.ndarray:
        if len(self.frames) == 1:
            return self.frames[0]

        total_ms = sum(self.frame_durations_ms)
        if total_ms <= 0:
            return self.frames[0]

        target_ms = int(elapsed_seconds * 1000) % total_ms
        acc = 0
        for frame, dur in zip(self.frames, self.frame_durations_ms):
            acc += dur
            if target_ms < acc:
                return frame
        return self.frames[-1]


def _draw_placeholder_badge(
    label: str,
    bg_color_bgr: tuple[int, int, int],
    subtext: str,
    size: int = 256,
) -> np.ndarray:
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    cv2.circle(canvas, center, radius, (*bg_color_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, (255, 255, 255, 255), 8, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15 if len(label) <= 5 else 0.9
    (tw, th), _ = cv2.getTextSize(label, font, scale, 4)
    tx = (size - tw) // 2
    ty = (size + th) // 2 - 12
    cv2.putText(canvas, label, (tx, ty), font, scale, (255, 255, 255, 255), 4, cv2.LINE_AA)

    (sw, _), _ = cv2.getTextSize(subtext, font, 0.65, 2)
    sx = (size - sw) // 2
    sy = ty + 38
    cv2.putText(canvas, subtext, (sx, sy), font, 0.65, (240, 240, 240, 255), 2, cv2.LINE_AA)

    return canvas


def ensure_placeholder_stickers() -> None:
    specs = {
        Path("assets/stickers/happy.png"): ("HAPPY", (40, 180, 255), ":D"),
        Path("assets/stickers/thumbs_up.png"): ("NICE!", (60, 180, 75), "GOOD JOB"),
        Path("assets/stickers/peace.png"): ("PEACE", (200, 150, 30), "V"),
        Path("assets/stickers/open_palm.png"): ("HELLO", (40, 110, 235), "HI!"),
        Path("assets/stickers/fist.png"): ("POW!", (180, 60, 160), "SOLID"),
    }

    for path, (label, color, subtext) in specs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            img = _draw_placeholder_badge(label, color, subtext)
            cv2.imwrite(str(path), img)
            print(f"Placeholder stiker dibuat: {path}")


def load_sticker_asset(path: Path, target_size: int = 160) -> StickerAsset:
    """Memuat PNG, WEBP (statis/animasi), atau GIF menjadi StickerAsset BGRA."""
    if not path.exists():
        raise FileNotFoundError(f"File stiker tidak ditemukan: {path}")

    frames: list[np.ndarray] = []
    durations: list[int] = []

    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            rgba = frame.convert("RGBA")
            arr_rgba = np.array(rgba, dtype=np.uint8)
            arr_bgra = cv2.cvtColor(arr_rgba, cv2.COLOR_RGBA2BGRA)

            h, w = arr_bgra.shape[:2]
            scale = target_size / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(arr_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frames.append(resized)
            durations.append(max(40, int(frame.info.get("duration") or 100)))

    if not frames:
        raise ValueError(f"Tidak ada frame yang terbaca dari stiker: {path}")

    return StickerAsset(frames=frames, frame_durations_ms=durations)