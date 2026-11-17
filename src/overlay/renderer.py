from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np

from src.overlay.sticker import StickerAsset, load_sticker_asset

Point = tuple[int, int]


@dataclass
class ActiveSticker:
    gesture: str
    asset: StickerAsset
    anchor_type: str
    last_position: Point
    started_at: float
    expires_at: float


def overlay_bgra(frame_bgr: np.ndarray, sticker_bgra: np.ndarray, center: Point) -> None:
    """Menempelkan stiker BGRA transparan pada titik tengah `center` di `frame_bgr`."""
    fh, fw = frame_bgr.shape[:2]
    sh, sw = sticker_bgra.shape[:2]

    x1 = center[0] - sw // 2
    y1 = center[1] - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(fw, x2)
    fy2 = min(fh, y2)

    if fx1 >= fx2 or fy1 >= fy2:
        return

    sx1 = fx1 - x1
    sy1 = fy1 - y1
    sx2 = sx1 + (fx2 - fx1)
    sy2 = sy1 + (fy2 - fy1)

    roi = frame_bgr[fy1:fy2, fx1:fx2]
    sticker_crop = sticker_bgra[sy1:sy2, sx1:sx2]

    alpha = sticker_crop[:, :, 3:4].astype(np.float32) / 255.0
    fg = sticker_crop[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)

    blended = fg * alpha + bg * (1.0 - alpha)
    frame_bgr[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)


class StickerOverlayEngine:
    def __init__(
        self,
        gesture_configs: dict[str, dict],
        display_duration: float = 2.0,
        cooldown_seconds: float = 3.0,
        sticker_size: int = 160,
        position_mode: str = "follow",
    ) -> None:
        self.display_duration = display_duration
        self.cooldown_seconds = cooldown_seconds
        self.sticker_size = sticker_size
        self.position_mode = position_mode

        self._sticker_pools: dict[str, list[StickerAsset]] = {}
        self._last_index: dict[str, int] = {}
        self._anchor_types: dict[str, str] = {}
        self._active: dict[str, ActiveSticker] = {}
        self._cooldown_until: dict[str, float] = {}

        for gesture_name, cfg in gesture_configs.items():
            raw_paths = cfg.get("stickers") or [cfg["sticker"]]
            assets = [
                load_sticker_asset(Path(p), target_size=sticker_size)
                for p in raw_paths
            ]
            self._sticker_pools[gesture_name] = assets
            self._anchor_types[gesture_name] = cfg.get("anchor", "hand_top")

    def _pick_asset(self, gesture_name: str) -> StickerAsset:
        pool = self._sticker_pools[gesture_name]
        if len(pool) == 1:
            return pool[0]

        prev_idx = self._last_index.get(gesture_name, -1)
        candidates = [i for i in range(len(pool)) if i != prev_idx]
        chosen_idx = random.choice(candidates)
        self._last_index[gesture_name] = chosen_idx
        return pool[chosen_idx]

    def trigger(self, gesture_name: str, position: Point, now: float | None = None) -> bool:
        if gesture_name not in self._sticker_pools:
            return False

        now = time.perf_counter() if now is None else now
        if gesture_name in self._active:
            return False
        if now < self._cooldown_until.get(gesture_name, 0.0):
            return False

        expires_at = now + self.display_duration
        self._active[gesture_name] = ActiveSticker(
            gesture=gesture_name,
            asset=self._pick_asset(gesture_name),
            anchor_type=self._anchor_types[gesture_name],
            last_position=position,
            started_at=now,
            expires_at=expires_at,
        )
        self._cooldown_until[gesture_name] = expires_at + self.cooldown_seconds
        print(f"[EVENT] Gesture '{gesture_name}' terdeteksi -> stiker muncul!")
        return True

    def update_and_render(
        self,
        frame_bgr: np.ndarray,
        live_positions: dict[str, Point],
        now: float | None = None,
    ) -> list[str]:
        now = time.perf_counter() if now is None else now
        fh, fw = frame_bgr.shape[:2]

        expired = [name for name, item in self._active.items() if now >= item.expires_at]
        for name in expired:
            del self._active[name]

        active_list = list(self._active.values())
        if self.position_mode == "corner" and active_list:
            active_list = [active_list[-1]]

        for item in active_list:
            if item.gesture in live_positions:
                item.last_position = live_positions[item.gesture]

            frame_bgra = item.asset.get_frame(now - item.started_at)

            if self.position_mode == "corner":
                sh, sw = frame_bgra.shape[:2]
                draw_pos = (fw - sw // 2 - 20, sh // 2 + 20)
            else:
                draw_pos = item.last_position

            overlay_bgra(frame_bgr, frame_bgra, draw_pos)

        return list(self._active.keys())

    def get_cooldown_names(self, now: float | None = None) -> list[str]:
        now = time.perf_counter() if now is None else now
        return [
            name
            for name, until in self._cooldown_until.items()
            if name not in self._active and now < until
        ]

    def clear(self) -> None:
        self._active.clear()