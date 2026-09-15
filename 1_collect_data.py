"""
جمع بيانات الإشارات (Data Collection)
=====================================
هذا السكربت يطابق قسم 4.1 (Data Collection: Ready-Made vs. Custom Datasets)
و 3.3.1 (MediaPipe) بالبحث.

الفكرة: بدل تصنيف يعتمد على قاعدة واحدة بسيطة (زي كود الملحق القديم)،
نسجل لكل إشارة عدد من "التسلسلات" (sequences) — كل تسلسل هو نافذة زمنية
من SEQUENCE_LENGTH فريم، وكل فريم فيه 21 نقطة (landmark) بثلاثة إحداثيات
(x, y, z) = 63 رقم لكل يد. هذا هو نفس التمثيل اللي يوصف بالفصل 4.2.2.

ملاحظة مهمة عن الترميز (Encoding):
    استخدم اسم الإشارة بحروف إنكليزية بسيطة (hello, thanks, yes ...)
    بدل العربي — هذا يفادي مشكلة علامات الاستفهام (????) اللي تظهر
    بـ Command Prompt الافتراضي بويندوز عند كتابة عربي بسطر الأوامر.
    الاسم العربي المعروض فعلياً على الشاشة وبالصوت يُقرأ من ملف
    translations.json (مثال: "hello" -> "مرحبا"). لو الاسم الإنكليزي
    غير موجود بالملف، النظام يستخدمه هو نفسه كما هو.

طريقة الاستخدام:
    python 1_collect_data.py --gesture hello --sequences 30
"""

import argparse
import json
import os

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False

ARABIC_FONT_PATH = "C:/Windows/Fonts/arial.ttf"
TRANSLATIONS_PATH = "translations.json"
_font_cache = None


def get_arabic_font(size: int = 32):
    global _font_cache
    if _font_cache is None:
        try:
            _font_cache = ImageFont.truetype(ARABIC_FONT_PATH, size)
        except Exception:
            _font_cache = ImageFont.load_default()
    return _font_cache


def draw_text_on_frame(frame, text: str, position=(30, 20), color=(0, 255, 0)):
    """نفس فكرة السكربت الثالث: نرسم بـ PIL حتى يظهر العربي متصل وبالاتجاه الصحيح."""
    display_text = text
    if ARABIC_SHAPING_AVAILABLE:
        try:
            display_text = get_display(arabic_reshaper.reshape(text))
        except Exception:
            display_text = text

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, display_text, font=get_arabic_font(34), fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_arabic_display_name(gesture_name: str) -> str:
    """يجيب الاسم العربي من translations.json لعرضه بس على الشاشة أثناء التسجيل."""
    if os.path.exists(TRANSLATIONS_PATH):
        with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
        return table.get(gesture_name, gesture_name)
    return gesture_name

SEQUENCE_LENGTH = 30      # عدد الفريمات بكل تسلسل (يطابق الوصف: نافذة زمنية للحركة)
NUM_HAND_LANDMARKS = 21   # ثابت حسب MediaPipe Hands
FEATURES_PER_FRAME = NUM_HAND_LANDMARKS * 3  # x,y,z لكل نقطة = 63

DATASET_DIR = "dataset"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(results):
    """
    يحول نتيجة MediaPipe إلى متجه ثابت الطول (63 رقم).
    إذا ما فيه يد بالفريم، يرجع أصفار حتى يضل طول المتجه ثابت
    (هذا مهم عشان الشبكة العصبية تحتاج مدخل بحجم موحّد).
    """
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        ).flatten()
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def collect(gesture_name: str, num_sequences: int, camera_index: int = 0):
    gesture_dir = os.path.join(DATASET_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing = len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
    arabic_name = load_arabic_display_name(gesture_name)

    cap = cv2.VideoCapture(camera_index)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        for seq_idx in range(existing, existing + num_sequences):
            window = []
            print(f"[{gesture_name}] تسجيل التسلسل رقم {seq_idx} ... جهّز الإشارة")

            # عد تنازلي بسيط قبل كل تسلسل حتى يجهز المستخدم
            for countdown in (3, 2, 1):
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame = draw_text_on_frame(
                    frame, f"استعد... {countdown} ({arabic_name})",
                    position=(20, 20), color=(0, 0, 255),
                )
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(500)

            while len(window) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                    )

                window.append(extract_landmarks(results))

                overlay = f"{arabic_name} | frame {len(window)}/{SEQUENCE_LENGTH}"
                frame = draw_text_on_frame(frame, overlay, position=(20, 20))
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            sequence_array = np.array(window, dtype=np.float32)  # shape: (30, 63)
            save_path = os.path.join(gesture_dir, f"seq_{seq_idx:03d}.npy")
            np.save(save_path, sequence_array)
            print(f"  -> حُفظ: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="جمع بيانات إشارات لغة الإشارة العربية")
    parser.add_argument(
        "--gesture", required=True,
        help="اسم الإشارة بحروف إنكليزية (مثال: hello) - يُستخدم كاسم مجلد وكـ class label",
    )
    parser.add_argument("--sequences", type=int, default=30, help="عدد التسلسلات المطلوب تسجيلها")
    parser.add_argument("--camera", type=int, default=0, help="رقم الكاميرا")
    args = parser.parse_args()

    collect(args.gesture, args.sequences, args.camera)
