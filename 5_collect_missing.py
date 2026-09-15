
الاستخدام:
    python 5_collect_missing.py
"""

import json
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except ImportError:
    ARABIC_OK = False

SEQUENCE_LENGTH   = 30
FEATURES_PER_FRAME = 63
DATASET_DIR       = "dataset"
TRANSLATIONS_PATH = "translations.json"
ARABIC_FONT_PATH  = "C:/Windows/Fonts/arial.ttf"
NUM_SEQUENCES     = 100  # عدد التسلسلات لكل كلمة - زد الرقم لدقة أعلى

_font_cache = None

def get_font(size=34):
    global _font_cache
    if _font_cache is None:
        try:
            _font_cache = ImageFont.truetype(ARABIC_FONT_PATH, size)
        except Exception:
            _font_cache = ImageFont.load_default()
    return _font_cache

def draw_text(frame, text, pos=(20, 20), color=(0, 255, 0)):
    disp = text
    if ARABIC_OK:
        try:
            disp = get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, disp, font=get_font(), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def extract_lm(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in hand.landmark],
                        dtype=np.float32).flatten()
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def collect_gesture(cap, hands, mp_hands, mp_draw, gesture_en, gesture_ar):
    gesture_dir = os.path.join(DATASET_DIR, gesture_en)
    os.makedirs(gesture_dir, exist_ok=True)
    existing = len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])

    for seq_idx in range(existing, existing + NUM_SEQUENCES):
        window = []

        # ── شاشة الانتظار ──────────────────────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            frame = draw_text(frame,
                f"التالي: {gesture_ar} ({gesture_en})  [{seq_idx+1}/{existing+NUM_SEQUENCES}]",
                pos=(20, 20), color=(0, 200, 255))
            frame = draw_text(frame,
                "اضغط SPACE للتسجيل  |  Q للخروج",
                pos=(20, 70), color=(180, 180, 180))
            cv2.imshow("Data Collection", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                waiting = False
            elif key == ord("q"):
                return False

        # ── عد تنازلي ──────────────────────────────────────────────────────
        for countdown in (3, 2, 1):
            for _ in range(10):   # ~300ms
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = draw_text(frame,
                    f"{gesture_ar}  -  {countdown}",
                    pos=(20, 20), color=(0, 0, 255))
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(30)

        # ── تسجيل التسلسل ──────────────────────────────────────────────────
        while len(window) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS)

            window.append(extract_lm(res))

            frame = draw_text(frame,
                f"{gesture_ar}  |  فريم {len(window)}/{SEQUENCE_LENGTH}",
                pos=(20, 20))
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

        arr = np.array(window, dtype=np.float32)
        path = os.path.join(gesture_dir, f"seq_{seq_idx:03d}.npy")
        np.save(path, arr)
        print(f"  ✓ حُفظ: {path}")

    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        translations = json.load(f)

    # حدد الكلمات الناقصة من الداتاسيت
    all_words = list(translations.keys())
    existing_folders = set(os.listdir(DATASET_DIR)) if os.path.exists(DATASET_DIR) else set()

    missing = [w for w in all_words if w not in existing_folders]
    partial = [w for w in all_words
               if w in existing_folders
               and len([f for f in os.listdir(os.path.join(DATASET_DIR, w))
                        if f.endswith(".npy")]) < NUM_SEQUENCES]

    to_collect = missing + partial

    if not to_collect:
        print("✅ جميع الكلمات موجودة في الداتاسيت بعدد كافٍ من التسلسلات!")
        return

    print(f"\n📋 الكلمات التي سيتم جمع بياناتها ({len(to_collect)}):")
    for i, w in enumerate(to_collect, 1):
        ar = translations.get(w, w)
        folder = os.path.join(DATASET_DIR, w)
        existing = len([f for f in os.listdir(folder) if f.endswith(".npy")]) \
                   if os.path.exists(folder) else 0
        print(f"  {i:2}. {w:12} ({ar}) - موجود: {existing}/{NUM_SEQUENCES}")

    print("\n🎥 جاهز للتسجيل - اضغط SPACE قبل كل إشارة ، Q للخروج\n")
    input("اضغط Enter للبدء...")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ تعذّر فتح الكاميرا")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        for idx, gesture_en in enumerate(to_collect, 1):
            gesture_ar = translations.get(gesture_en, gesture_en)
            print(f"\n[{idx}/{len(to_collect)}] 📌 {gesture_en} ({gesture_ar})")
            ok = collect_gesture(cap, hands, mp_hands, mp_draw,
                                 gesture_en, gesture_ar)
            if not ok:
                print("⚠️  تم الخروج المبكر")
                break
            print(f"  ✅ اكتملت '{gesture_ar}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ انتهى جمع البيانات!")
    print("▶️  شغّل التدريب الآن:")
    print("    python 2_train_model.py")


if __name__ == "__main__":
    main()
