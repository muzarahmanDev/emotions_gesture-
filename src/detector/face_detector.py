from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


@dataclass
class FaceDetectionResult:
    landmarks: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    smile_score: float = 0.0


def ensure_face_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh model MediaPipe FaceLandmarker ke {model_path}...")
    urllib.request.urlretrieve(FACE_MODEL_URL, model_path)
    print("Model berhasil diunduh.")
    return model_path


class FaceDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        max_faces: int = 1,
        min_confidence: float = 0.5,
    ) -> None:
        resolved_path = ensure_face_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(resolved_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetectionResult]:
        h, w, _ = frame_bgr.shape

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections: list[FaceDetectionResult] = []
        for idx, face_landmarks in enumerate(result.face_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            pad = 15
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w - 1, max(xs) + pad)
            y2 = min(h - 1, max(ys) + pad)

            smile_score = 0.0
            if result.face_blendshapes and idx < len(result.face_blendshapes):
                scores = {b.category_name: b.score for b in result.face_blendshapes[idx]}
                smile_score = (
                    scores.get("mouthSmileLeft", 0.0)
                    + scores.get("mouthSmileRight", 0.0)
                ) / 2.0

            detections.append(
                FaceDetectionResult(
                    landmarks=points,
                    bbox=(x1, y1, x2, y2),
                    smile_score=smile_score,
                )
            )

        return detections

    def close(self) -> None:
        self._landmarker.close()


def draw_face_detections(
    frame_bgr: np.ndarray,
    detections: list[FaceDetectionResult],
) -> np.ndarray:
    for face in detections:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"Face (Smile: {face.smile_score:.2f})",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        for x, y in face.landmarks:
            cv2.circle(frame_bgr, (x, y), 1, (255, 255, 0), -1)

    return frame_bgr