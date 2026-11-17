from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreviewAction:
    quit_requested: bool = False
    toggle_effects: bool = False
    toggle_landmarks: bool = False


class PreviewWindow:
    def __init__(self, window_name: str = "emotions_gesture - Preview") -> None:
        self.window_name = window_name

    def render_hud(
        self,
        frame_bgr: np.ndarray,
        fps: float,
        effects_enabled: bool,
        show_landmarks: bool,
        active_stickers: list[str],
        vcam_name: str | None = None,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        if effects_enabled:
            badge_color = (40, 170, 60)
            status_text = "ON | EFFECTS ACTIVE"
        else:
            badge_color = (50, 50, 210)
            status_text = "OFF | PAUSED (NORMAL)"

        cv2.rectangle(frame_bgr, (16, 14), (285, 52), badge_color, -1)
        cv2.putText(
            frame_bgr,
            status_text,
            (26, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        info_parts = [f"FPS: {fps:.1f}"]
        if vcam_name:
            info_parts.append(f"VCam: {vcam_name}")
        if effects_enabled and active_stickers:
            info_parts.append(f"Active: {', '.join(active_stickers)}")

        cv2.putText(
            frame_bgr,
            " | ".join(info_parts),
            (300, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0) if effects_enabled else (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        lm_state = "ON" if show_landmarks else "OFF"
        hint = f"[SPACE / T] Toggle Effects   [L] Landmarks: {lm_state}   [Q / ESC] Quit"
        cv2.rectangle(frame_bgr, (0, h - 34), (w, h), (20, 20, 20), -1)
        cv2.putText(
            frame_bgr,
            hint,
            (16, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return frame_bgr

    def show_and_poll(self, frame_bgr: np.ndarray) -> PreviewAction:
        cv2.imshow(self.window_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        action = PreviewAction()
        if key in (ord("q"), ord("Q"), 27):
            action.quit_requested = True
        elif key in (32, ord("t"), ord("T")):  # 32 = tombol SPACE
            action.toggle_effects = True
        elif key in (ord("l"), ord("L")):
            action.toggle_landmarks = True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            action.quit_requested = True

        return action

    def close(self) -> None:
        cv2.destroyAllWindows()