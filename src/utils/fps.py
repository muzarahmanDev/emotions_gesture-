import time


class FPSCounter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self._prev_time: float | None = None
        self.fps: float = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._prev_time is None:
            self._prev_time = now
            return self.fps

        dt = now - self._prev_time
        self._prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (
                    self.smoothing * self.fps
                    + (1.0 - self.smoothing) * instant_fps
                )

        return self.fps