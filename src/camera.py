import cv2
import numpy as np


class WebcamCamera:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Tidak bisa membuka webcam (index={self.camera_index}). "
                "Pastikan kamera tidak sedang dipakai aplikasi lain."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        success, frame = self.cap.read()
        if not success:
            return None
        return frame 

    def get_resolution(self) -> tuple[int, int]:
        if self.cap is None:
            return self.width, self.height
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        return actual_w, actual_h

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None