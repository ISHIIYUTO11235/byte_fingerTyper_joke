"""
The Byte Weaver - ハンドトラッキングによるASCII文字入力アプリケーション
(完全版: Joy-Conグリップ仕様 + 顔検出ロック + 超高速キャッシュ描画 + 遅延無効化)
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Tuple, Optional, List
import os

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# 設定定数

TOUCH_THRESHOLD_RATIO = 0.15  # 手首〜親指ラインに対する接触判定の
ASCII_TABLE_WIDTH = 400

# ★ PyAutoGUIのフェイルセーフ遅延を無効化（入力時のフリーズを解消しFPSを最大化）
pyautogui.PAUSE = 0


# MediaPipe 初期化

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)


# ビットマッピング定義

BIT_MAPPING = [
    ("Right", 20, 0),  # Bit 1 (LSB): 右手 小指
    ("Right", 16, 1),  # Bit 2: 右手 薬指
    ("Right", 12, 2),  # Bit 3: 右手 中指
    ("Right", 8, 3),   # Bit 4: 右手 人差し指
    ("Left", 8, 4),    # Bit 5: 左手 人差し指
    ("Left", 12, 5),   # Bit 6: 左手 中指
    ("Left", 16, 6),   # Bit 7 (MSB): 左手 薬指
]

THUMB_JOINTS = [0, 1, 2, 3, 4]  # 手首(0)から親指の先端(4)
WRIST = 0
MIDDLE_MCP = 9


# フォント読み込み設定

FONT_BIT_CANDIDATES = [
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consola.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "cour.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "lucon.ttf"),
]
FONT_ASCII_CANDIDATES = [
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "segoeui.ttf"),
    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consola.ttf"),
]

_font_bit_cache = {}
_font_ascii_cache = {}

def _load_font(candidates: List[str], size: int):
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def _get_bit_font(size: int = 28):
    global _font_bit_cache
    if size not in _font_bit_cache:
        _font_bit_cache[size] = _load_font(FONT_BIT_CANDIDATES, size)
    return _font_bit_cache[size]

def _get_ascii_font(size: int = 16):
    global _font_ascii_cache
    if size not in _font_ascii_cache:
        _font_ascii_cache[size] = _load_font(FONT_ASCII_CANDIDATES, size)
    return _font_ascii_cache[size]

def _draw_text_pil(canvas_bgr: np.ndarray, text: str, xy: Tuple[int, int], font,
                   color_bgr: Tuple[int, int, int], outline_bgr: Optional[Tuple[int, int, int]] = None) -> None:
    if not HAS_PIL:
        cv2.putText(canvas_bgr, text, (xy[0], xy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return
    pil_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    if outline_bgr:
        outline_rgb = (outline_bgr[2], outline_bgr[1], outline_bgr[0])
        draw.text(xy, text, font=font, fill=color_rgb, outline=outline_rgb, width=1)
    else:
        draw.text(xy, text, font=font, fill=color_rgb)
    out = np.array(pil_img)
    canvas_bgr[:, :] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ==========================================
# 数学・距離計算ユーティリティ
# ==========================================
def get_pixel_coords(landmark, w: int, h: int) -> Tuple[float, float]:
    return (landmark.x * w, landmark.y * h)

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def point_to_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    ab = np.array([bx - ax, by - ay])
    ap = np.array([px - ax, py - ay])
    length_sq = np.dot(ab, ab)
    if length_sq == 0.0:
        return calculate_distance(p, a)
    t = max(0, min(1, np.dot(ap, ab) / length_sq))
    projection = np.array([ax, ay]) + t * ab
    return np.linalg.norm(np.array([px, py]) - projection)

def get_reference_length(landmarks, w: int, h: int) -> float:
    wrist = get_pixel_coords(landmarks.landmark[WRIST], w, h)
    middle_mcp = get_pixel_coords(landmarks.landmark[MIDDLE_MCP], w, h)
    return calculate_distance(wrist, middle_mcp)

# ==========================================
# 接触判定ロジック
# ==========================================
def is_touching_thumb_line(finger_tip: Tuple[float, float], thumb_points: List[Tuple[float, float]], ref_length: float) -> bool:
    min_dist = float('inf')
    for i in range(len(thumb_points) - 1):
        dist = point_to_segment_distance(finger_tip, thumb_points[i], thumb_points[i+1])
        if dist < min_dist:
            min_dist = dist
    threshold = ref_length * TOUCH_THRESHOLD_RATIO
    return min_dist <= threshold

def get_bit_value(hand_landmarks, finger_tip_landmark: int, ref_length: float, w: int, h: int) -> bool:
    thumb_points = [get_pixel_coords(hand_landmarks.landmark[i], w, h) for i in THUMB_JOINTS]
    finger_tip = get_pixel_coords(hand_landmarks.landmark[finger_tip_landmark], w, h)
    return is_touching_thumb_line(finger_tip, thumb_points, ref_length)

def read_binary_from_hands(multi_hand_landmarks, multi_handedness, w: int, h: int) -> Tuple[int, List[bool], dict]:
    bits = [False] * 7
    ref_lengths = {}
    hands_dict = {}
    
    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        label = multi_handedness[idx].classification[0].label
        hands_dict[label] = hand_landmarks
        ref_lengths[label] = get_reference_length(hand_landmarks, w, h)
    
    for hand_label, finger_tip, bit_pos in BIT_MAPPING:
        if hand_label not in hands_dict:
            continue
        hand_landmarks = hands_dict[hand_label]
        ref_length = ref_lengths[hand_label]
        bits[bit_pos] = get_bit_value(hand_landmarks, finger_tip, ref_length, w, h)
    
    ascii_val = sum(bits[i] * (2 ** i) for i in range(7))
    return ascii_val, bits, ref_lengths

def check_trigger(multi_hand_landmarks, multi_handedness, w: int, h: int) -> bool:
    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        label = multi_handedness[idx].classification[0].label
        if label == "Left":
            ref_length = get_reference_length(hand_landmarks, w, h)
            thumb_points = [get_pixel_coords(hand_landmarks.landmark[i], w, h) for i in THUMB_JOINTS]
            pinky_tip = get_pixel_coords(hand_landmarks.landmark[20], w, h)
            return is_touching_thumb_line(pinky_tip, thumb_points, ref_length)
    return False

#
# UI 描画 (カメラ側)

def draw_hand_landmarks(image, multi_hand_landmarks, multi_handedness, bits: List[bool], ref_lengths: dict, is_trigger_pulled: bool):
    if multi_hand_landmarks is None:
        return image
    
    h, w, _ = image.shape
    
    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        label = multi_handedness[idx].classification[0].label
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # 親指ライン(Joy-Con判定エリア)の強調描画
        for i in range(len(THUMB_JOINTS) - 1):
            pt1 = get_pixel_coords(hand_landmarks.landmark[THUMB_JOINTS[i]], w, h)
            pt2 = get_pixel_coords(hand_landmarks.landmark[THUMB_JOINTS[i+1]], w, h)
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 200, 0), 4)
        
        ref_length = ref_lengths.get(label, 0)
        circle_radius = max(5, int(ref_length * TOUCH_THRESHOLD_RATIO))
        
        for hand_label, finger_tip_landmark, bit_pos in BIT_MAPPING:
            if hand_label == label:
                finger_tip = hand_landmarks.landmark[finger_tip_landmark]
                cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)
                color = (0, 255, 0) if bits[bit_pos] else (0, 0, 255)
                cv2.circle(image, (cx, cy), circle_radius, color, -1)
        
        if label == "Left":
            pinky_tip = hand_landmarks.landmark[20]
            cx, cy = int(pinky_tip.x * w), int(pinky_tip.y * h)
            trigger_color = (255, 0, 255) if is_trigger_pulled else (128, 128, 128)
            cv2.circle(image, (cx, cy), circle_radius + 2, trigger_color, 2)
    
    return image

def draw_overlay_info(image, bits: List[bool], ascii_val: int, char: str, is_active: bool):
    h, w, _ = image.shape
    
    if not is_active:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 50), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        cv2.putText(image, "SYSTEM LOCKED", (w//2 - 150, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(image, "NO FACE DETECTED", (w//2 - 120, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        return image

    bit_string = ''.join(['1' if bits[i] else '0' for i in range(6, -1, -1)])
    info_text = f"Bits: {bit_string}"
    char_text = f"ASCII: {ascii_val} = '{char}'"
    
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    font_bit = _get_bit_font(28)
    font_ascii_line = _get_ascii_font(18)
    _draw_text_pil(image, info_text, (20, 18), font_bit, (0, 255, 0), outline_bgr=(0, 80, 0))
    _draw_text_pil(image, char_text, (20, 58), font_ascii_line, (0, 255, 255), outline_bgr=(0, 80, 80))
    
    return image

# ==========================================
# 超高速化：ASCIIテーブル描画ロジック (キャッシュ処理)
# ==========================================
_cached_bg_active = None
_cached_bg_locked = None

def _generate_static_ascii_bg(width: int, actual_height: int, chars_per_row: int, cell_width: int, cell_height: int, is_active: bool) -> np.ndarray:
    """ASCIIテーブルの静的背景（文字のみ）を1回だけ生成する"""
    bg = np.zeros((actual_height, width, 3), dtype=np.uint8)
    
    if not HAS_PIL:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for ascii_val in range(32, 127):
            row = (ascii_val - 32) // chars_per_row
            col = (ascii_val - 32) % chars_per_row
            x = col * cell_width
            y = row * cell_height
            char = chr(ascii_val)
            text_color = (255, 255, 255) if is_active else (100, 100, 100)
            cv2.putText(bg, char, (x + 5, y + 20), font, 0.5, text_color, 1)
        return bg

    font_char = _get_ascii_font(18)
    font_code = _get_ascii_font(10)
    pil_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    for ascii_val in range(32, 127):
        row = (ascii_val - 32) // chars_per_row
        col = (ascii_val - 32) % chars_per_row
        x = col * cell_width
        y = row * cell_height
        char = chr(ascii_val)
        
        text_color = (255, 255, 255) if is_active else (100, 100, 100)
        code_color = (180, 180, 180) if is_active else (80, 80, 80)
        
        try:
            char_bbox = draw.textbbox((0, 0), char, font=font_char)
        except AttributeError:
            char_bbox = (0, 0, cell_width, cell_height)
        cw = char_bbox[2] - char_bbox[0]
        ch = char_bbox[3] - char_bbox[1]
        tx = x + (cell_width - cw) // 2
        ty = y + (cell_height - ch) // 2 - 2
        
        draw.text((tx, ty), char, font=font_char, fill=text_color)
        draw.text((x + 2, y + 2), str(ascii_val), font=font_code, fill=code_color)
        
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_ascii_table(current_ascii_val: int, width: int = ASCII_TABLE_WIDTH, height: int = 600, is_active: bool = True) -> np.ndarray:
    global _cached_bg_active, _cached_bg_locked
    
    chars_per_row = 16
    cell_width = width // chars_per_row
    cell_height = 28
    
    num_rows = ((127 - 32) + chars_per_row - 1) // chars_per_row
    actual_height = num_rows * cell_height
    if actual_height > height:
        actual_height = height
        
    # 初回実行時のみキャッシュを生成
    if _cached_bg_active is None:
        _cached_bg_active = _generate_static_ascii_bg(width, actual_height, chars_per_row, cell_width, cell_height, True)
        _cached_bg_locked = _generate_static_ascii_bg(width, actual_height, chars_per_row, cell_width, cell_height, False)

    # 毎フレームの処理：キャッシュ画像をコピー（速）
    table_image = _cached_bg_active.copy() if is_active else _cached_bg_locked.copy()
        
    # アクティブ時のみ、現在のASCII値に黄色いハイライトをOpenCVで重ねる（高速）
    if is_active and 32 <= current_ascii_val <= 126:
        row = (current_ascii_val - 32) // chars_per_row
        col = (current_ascii_val - 32) % chars_per_row
        x = col * cell_width
        y = row * cell_height
        
        overlay = table_image.copy()
        cv2.rectangle(overlay, (x, y), (x + cell_width - 1, min(y + cell_height - 1, actual_height - 1)), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, table_image, 0.6, 0, table_image)
        
        # ハイライトされた文字だけ黒色で上書き描画して視認性を高める
        char = chr(current_ascii_val)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(char, font, 0.6, 2)[0]
        tx = x + (cell_width - text_size[0]) // 2
        ty = y + (cell_height + text_size[1]) // 2
        cv2.putText(table_image, char, (tx, ty), font, 0.6, (0, 0, 0), 2)
        
    return table_image


# メインループ

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return
    
    is_trigger_pulled_prev = False
    
    print("The Byte Weaver を起動しました")
    print("'q'キーで終了します")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 顔検出
        face_results = face_detection.process(rgb_frame)
        is_active = face_results.detections is not None

        ascii_val = 0
        bits = [False] * 7
        is_trigger_pulled = False
        
        if is_active:
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                ascii_val, bits, ref_lengths = read_binary_from_hands(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                    w, h
                )
                is_trigger_pulled = check_trigger(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                    w, h
                )
                frame = draw_hand_landmarks(
                    frame,
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                    bits,
                    ref_lengths,
                    is_trigger_pulled
                )
        
        char = chr(ascii_val) if 32 <= ascii_val <= 126 else '?'
        
        frame = draw_overlay_info(frame, bits, ascii_val, char, is_active)
        
        cam_height = frame.shape[0]
        ascii_table = create_ascii_table(ascii_val, height=cam_height, is_active=is_active)
        
        if ascii_table.shape[0] != cam_height:
            ascii_table_resized = cv2.resize(ascii_table, (ASCII_TABLE_WIDTH, cam_height))
        else:
            ascii_table_resized = ascii_table
        
        combined_frame = cv2.hconcat([frame, ascii_table_resized])
        cv2.imshow('The Byte Weaver', combined_frame)
        
        # トリガー処理
        if is_active and is_trigger_pulled and not is_trigger_pulled_prev:
            if 32 <= ascii_val <= 126:
                try:
                    pyautogui.write(char)
                except Exception:
                    pass
            elif ascii_val == 8:
                try:
                    pyautogui.press('backspace')
                except Exception:
                    pass
            elif ascii_val == 13:
                try:
                    pyautogui.press('enter')
                except Exception:
                    pass
        
        is_trigger_pulled_prev = is_trigger_pulled if is_active else False
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_detection.close()

if __name__ == "__main__":
    main()