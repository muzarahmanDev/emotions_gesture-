from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

from src.detector.gesture_rules import GESTURE_LABELS, classify_hand_gesture

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_HAND_MODEL_PATH = Path("models/hand_landmarker.task")

# Pasangan indeks sendi untuk menggambar garis tulang tangan
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (0, 9), (9, 10), (10, 11), (11, 12),   # Jari tengah
    (0, 13), (13, 14), (14, 15), (15, 16), # Jari manis
    (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking
    (5, 9), (9, 13), (13, 17),             # Telapak tangan
)


@dataclass
class HandDetectionResult:
    landmarks: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    handedness: str
    gesture: str | None


def ensure_hand_model(model_path: Path = DEFAULT_HAND_MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh model MediaPipe HandLandmarker ke {model_path}...")
    urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    print("Model HandLandmarker berhasil diunduh.")
    return model_path


class HandDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_HAND_MODEL_PATH,
        max_hands: int = 2,
        min_confidence: float = 0.5,
    ) -> None:
        resolved_path = ensure_hand_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(resolved_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetectionResult]:
        h, w, _ = frame_bgr.shape

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections: list[HandDetectionResult] = []
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            pad = 15
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w - 1, max(xs) + pad)
            y2 = min(h - 1, max(ys) + pad)

            handedness = "Hand"
            if idx < len(result.handedness) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name

            gesture = classify_hand_gesture(points)
            detections.append(
                HandDetectionResult(
                    landmarks=points,
                    bbox=(x1, y1, x2, y2),
                    handedness=handedness,
                    gesture=gesture,
                )
            )

        return detections

    def close(self) -> None:
        self._landmarker.close()


def draw_hand_detections(
    frame_bgr: np.ndarray,
    detections: list[HandDetectionResult],
) -> np.ndarray:
    for hand in detections:
        x1, y1, x2, y2 = hand.bbox

        # Gambar garis tulang tangan
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                frame_bgr,
                hand.landmarks[start_idx],
                hand.landmarks[end_idx],
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        # Gambar 21 titik sendi tangan
        for x, y in hand.landmarks:
            cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)

        # Gambar kotak pembatas & label gesture
        color = (0, 220, 255) if hand.gesture else (200, 200, 200)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = GESTURE_LABELS.get(hand.gesture, "Unknown") if hand.gesture else "Hand"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame_bgr