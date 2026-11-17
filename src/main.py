import time

import cv2

from src.camera import WebcamCamera
from src.detector.face_detector import FaceDetector, draw_face_detections
from src.detector.hand_detector import HandDetector, draw_hand_detections
from src.output.preview import PreviewWindow
from src.output.virtualcam import VirtualCameraOutput
from src.overlay.renderer import StickerOverlayEngine
from src.overlay.sticker import ensure_placeholder_stickers
from src.utils.config import load_config
from src.utils.fps import FPSCounter


def main() -> None:
    config = load_config()
    ensure_placeholder_stickers()

    cam_cfg = config["camera"]
    vcam_cfg = config.get("virtualcam", {})
    ov_cfg = config["overlay"]
    smile_threshold = float(config.get("smile", {}).get("threshold", 0.45))
    sticker_size = int(ov_cfg.get("sticker_size", 160))

    effects_enabled = bool(ov_cfg.get("enabled_on_start", True))
    show_landmarks = bool(ov_cfg.get("show_debug_landmarks", True))
    vcam_enabled = bool(vcam_cfg.get("enabled", True))
    vcam_fps = int(vcam_cfg.get("fps", 30))

    camera = WebcamCamera(
        camera_index=int(cam_cfg.get("index", 0)),
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
    )
    face_detector = FaceDetector(max_faces=1)
    hand_detector = HandDetector(max_hands=2)
    overlay_engine = StickerOverlayEngine(
        gesture_configs=config["gestures"],
        display_duration=float(ov_cfg.get("display_duration", 2.0)),
        cooldown_seconds=float(ov_cfg.get("cooldown_seconds", 3.0)),
        sticker_size=sticker_size,
        position_mode=str(ov_cfg.get("position_mode", "follow")),
    )
    preview = PreviewWindow("emotions_gesture - Preview")
    fps_counter = FPSCounter()
    vcam: VirtualCameraOutput | None = None

    try:
        camera.open()
        if vcam_enabled:
            actual_w, actual_h = camera.get_resolution()
            vcam = VirtualCameraOutput(width=actual_w, height=actual_h, fps=vcam_fps)
            vcam.open()

        print("Aplikasi berjalan. Tekan SPACE/T untuk ON/OFF, L untuk landmarks, Q/ESC untuk keluar.")

        while True:
            frame = camera.read()
            if frame is None:
                print("Gagal membaca frame dari webcam.")
                break

            if cam_cfg.get("mirror", True):
                frame = cv2.flip(frame, 1)

            clean_frame = frame
            faces = []
            hands = []
            active_stickers: list[str] = []

            if effects_enabled:
                now = time.perf_counter()
                faces = face_detector.detect(clean_frame)
                hands = hand_detector.detect(clean_frame)

                live_positions: dict[str, tuple[int, int]] = {}
                half_sticker = sticker_size // 2

                for face in faces:
                    x1, y1, x2, _ = face.bbox
                    cx = (x1 + x2) // 2
                    cy = max(half_sticker + 10, y1 - half_sticker - 10)
                    live_positions["smile"] = (cx, cy)

                    if face.smile_score >= smile_threshold:
                        overlay_engine.trigger("smile", (cx, cy), now=now)

                for hand in hands:
                    if hand.gesture is None:
                        continue
                    x1, y1, x2, _ = hand.bbox
                    cx = (x1 + x2) // 2
                    cy = max(half_sticker + 10, y1 - half_sticker - 10)
                    live_positions[hand.gesture] = (cx, cy)
                    overlay_engine.trigger(hand.gesture, (cx, cy), now=now)

                active_stickers = overlay_engine.update_and_render(
                    clean_frame, live_positions, now=now
                )

            # Kirim frame bersih (hanya video + stiker) ke Virtual Camera
            if vcam is not None:
                vcam.send(clean_frame)

            # Preview lokal tetap menampilkan landmark (jika ON) & HUD kontrol
            preview_frame = clean_frame.copy()
            if effects_enabled and show_landmarks:
                draw_face_detections(preview_frame, faces)
                draw_hand_detections(preview_frame, hands)

            fps = fps_counter.update()
            preview.render_hud(
                preview_frame,
                fps=fps,
                effects_enabled=effects_enabled,
                show_landmarks=show_landmarks,
                active_stickers=active_stickers,
                vcam_name=vcam.device_name if vcam else None,
            )

            action = preview.show_and_poll(preview_frame)
            if action.quit_requested:
                break
            if action.toggle_effects:
                effects_enabled = not effects_enabled
                if not effects_enabled:
                    overlay_engine.clear()
                print(f"[TOGGLE] Effects {'ENABLED' if effects_enabled else 'DISABLED'}")
            if action.toggle_landmarks:
                show_landmarks = not show_landmarks
                print(f"[TOGGLE] Landmarks {'ON' if show_landmarks else 'OFF'}")

    finally:
        if vcam is not None:
            vcam.close()
        hand_detector.close()
        face_detector.close()
        camera.release()
        preview.close()
        print("Aplikasi ditutup dengan aman.")


if __name__ == "__main__":
    main()