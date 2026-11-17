import math

Point = tuple[int, int]

# Indeks landmark jari (PIP/sendi tengah, TIP/ujung jari)
FINGER_JOINTS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}

GESTURE_LABELS = {
    "open_palm": "OPEN PALM",
    "thumbs_up": "THUMBS UP",
    "peace": "PEACE",
    "fist": "FIST",
}


def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _is_finger_extended(landmarks: list[Point], pip_idx: int, tip_idx: int) -> bool:
    wrist = landmarks[0]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]
    return _distance(wrist, tip) > _distance(wrist, pip) * 1.12


def classify_hand_gesture(landmarks: list[Point]) -> str | None:
    """Mengklasifikasikan 21 landmark tangan menjadi nama gesture sederhana."""
    if len(landmarks) < 21:
        return None

    # Referensi skala ukuran tangan: jarak pergelangan (0) ke pangkal jari tengah (9)
    hand_size = _distance(landmarks[0], landmarks[9])
    if hand_size < 15:
        return None

    index_ext = _is_finger_extended(landmarks, *FINGER_JOINTS["index"])
    middle_ext = _is_finger_extended(landmarks, *FINGER_JOINTS["middle"])
    ring_ext = _is_finger_extended(landmarks, *FINGER_JOINTS["ring"])
    pinky_ext = _is_finger_extended(landmarks, *FINGER_JOINTS["pinky"])

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    # Cek apakah jempol terbuka menjauh dari telapak tangan
    thumb_out = (
        _distance(thumb_tip, pinky_mcp) > _distance(thumb_ip, pinky_mcp) * 1.05
        and _distance(thumb_tip, index_mcp) > 0.35 * hand_size
    )

    # 1. OPEN PALM: semua 4 jari lurus + jempol terbuka
    if index_ext and middle_ext and ring_ext and pinky_ext and thumb_out:
        return "open_palm"

    # 2. PEACE: telunjuk & tengah lurus, manis & kelingking menekuk
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    # 3. THUMBS UP: 4 jari menekuk + jempol menunjuk ke atas (ingat: y kecil = lebih atas)
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        thumb_pointing_up = (
            thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
            and (thumb_mcp[1] - thumb_tip[1]) > 0.35 * hand_size
            and _distance(thumb_tip, index_mcp) > 0.45 * hand_size
        )
        if thumb_pointing_up:
            return "thumbs_up"

        # 4. FIST: 4 jari menekuk dan jempol terlipat dekat jari
        if _distance(thumb_tip, index_mcp) < 0.65 * hand_size:
            return "fist"

    return None