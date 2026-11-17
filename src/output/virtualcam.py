import cv2
import numpy as np
import pyvirtualcam


class VirtualCameraOutput:
    def __init__(self, width: int, height: int, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._cam: pyvirtualcam.Camera | None = None

    @property
    def device_name(self) -> str:
        if self._cam is None:
            return "OFF"
        return self._cam.device

    def open(self) -> None:
        try:
            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Gagal membuka Virtual Camera. Pastikan OBS Studio sudah terinstall "
                "di Windows dan Virtual Camera di dalam aplikasi OBS tidak sedang aktif.\n"
                f"Detail error: {exc}"
            ) from exc

        print(
            f"Virtual Camera aktif: '{self._cam.device}' "
            f"({self.width}x{self.height} @ {self.fps} FPS)"
        )

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._cam is None:
            return

        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        self._cam.send(frame_bgr)

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None