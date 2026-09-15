"""
الترجمة الحية (Real-Time Sign-to-Speech)
=========================================
يطابق قسم 4.2 (System Stages: From Input to Voice Output) بالكامل:
    4.2.1 Input            -> قراءة الكاميرا بـ OpenCV
    4.2.2 Detection        -> MediaPipe لاستخراج 21 نقطة لليد
    4.2.3 Classification   -> نموذج CNN-LSTM المدرّب (2_train_model.py)
    4.2.4 Output           -> تحويل النص لصوت عبر gTTS (نفس فكرة appendix
                               الأصلي، لكن الآن التصنيف حقيقي مو rule-based)

الفرق عن كود الملحق القديم: بدل قاعدة واحدة بسيطة
(if lm.landmark[8].y < lm.landmark[6].y) صار عندنا نافذة منزلقة
(sliding window) من 30 فريم تتغذى فيها الشبكة CNN-LSTM المدربة فعلياً،
وهذا يطابق الدقة المذكورة بقسم 5.1 (Static ~95%, Dynamic ~89%).

عن الترجمة العربية:
    النموذج مدرّب على تسميات إنكليزية (hello, thanks ...) لأنها هي
    اسم المجلد اللي جمعنا منه البيانات بـ 1_collect_data.py. هذا
    السكربت يقرأ translations.json ويحوّل التسمية الإنكليزية لنص
    عربي قبل ما يعرضه على الشاشة أو ينطقه صوتياً. عدّل الملف حسب
    الإشارات اللي جمعتها فعلاً.

الاستخدام:
    python 3_realtime_translate.py
"""

import sys
import json
import os

# Fix UnicodeEncodeError in Windows CMD
sys.stdout.reconfigure(encoding='utf-8')
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

try:
    from gtts import gTTS
    from playsound3 import playsound
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False

# خط يدعم عربي — بويندوز عادة موجود بهذا المسار. غيّره إذا جهازك مختلف
# (لينكس مثلاً: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" ما يدعم عربي،
# جرب تحمّل خط عربي وتضع مساره هنا).
ARABIC_FONT_PATH = "C:/Windows/Fonts/arial.ttf"
_font_cache = None


def get_arabic_font(size: int = 32):
    global _font_cache
    if _font_cache is None:
        try:
            _font_cache = ImageFont.truetype(ARABIC_FONT_PATH, size)
        except Exception:
            _font_cache = ImageFont.load_default()
    return _font_cache


def draw_text_on_frame(frame, text: str, position=(30, 20)):
    """
    يرسم نص (عربي أو إنكليزي) على الفريم عبر PIL بدل cv2.putText،
    لأن cv2.putText ما يدعم رسم حروف عربية متصلة أو اتجاه RTL بشكل صحيح.
    لو مكتبتا arabic_reshaper/python-bidi متوفرين، نستخدمهم لتشكيل
    الحروف وترتيبها بالاتجاه الصحيح قبل الرسم.
    """
    display_text = text
    if ARABIC_SHAPING_AVAILABLE:
        try:
            reshaped = arabic_reshaper.reshape(text)
            display_text = get_display(reshaped)
        except Exception:
            display_text = text

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, display_text, font=get_arabic_font(34), fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

MODEL_PATH = "arsl_model.h5"
LABELS_PATH = "labels.json"
TRANSLATIONS_PATH = "translations.json"

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63
CONFIDENCE_THRESHOLD = 0.50   # لا نعلن إشارة إلا إذا كانت ثقة النموذج فوق هالحد


def load_translations():
    """
    يقرأ translations.json لو موجود، ويرجع قاموس {English: Arabic}.
    لو الملف غير موجود، يرجع قاموس فاضي (النظام وقتها يعرض التسمية
    الإنكليزية كما هي بدون كسر).
    """
    if os.path.exists(TRANSLATIONS_PATH):
        with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        ).flatten()
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def speak_async(text: str, lang: str = "ar"):
    """
    يحاول النطق الصوتي عبر gTTS (أونلاين) أولاً،
    ثم pyttsx3 (أوفلاين) كبديل، وإلا يطبع النص فقط.
    """
    print(f"[إشارة مكتشفة] {text}")

    def _run():
        # محاولة 1: gTTS (يحتاج إنترنت)
        if TTS_AVAILABLE:
            try:
                tmp_file = "tts_output.mp3"
                gTTS(text=text, lang=lang).save(tmp_file)
                playsound(tmp_file)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                return
            except Exception as exc:
                print(f"[gTTS فشل] {exc} - جرب pyttsx3")

        # محاولة 2: pyttsx3 (لا يحتاج إنترنت)
        if PYTTSX3_AVAILABLE:
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
                return
            except Exception as exc:
                print(f"[pyttsx3 فشل] {exc}")

        print(f"[لا يوجد صوت] الإشارة: {text}")

    threading.Thread(target=_run, daemon=True).start()


def main(camera_index: int = 0):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"ما لقيت النموذج {MODEL_PATH}. شغّل 2_train_model.py أول شي."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        gesture_labels = json.load(f)  # تسميات إنكليزية (نفس أسماء مجلدات dataset/)

    translations = load_translations()

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_spoken = ""

    cap = cv2.VideoCapture(camera_index)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
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

            frame_buffer.append(extract_landmarks(results))

            detected_text = ""
            confidence = 0.0
            raw_label = ""

            buf_size = len(frame_buffer)
            if buf_size == SEQUENCE_LENGTH:
                window = np.array(frame_buffer, dtype=np.float32)

                # Normalization: subtract wrist and divide by scale
                base = None
                scale = 1.0
                for i in range(SEQUENCE_LENGTH):
                    if np.any(window[i]):
                        base = window[i, 0:3]
                        pts = window[i].reshape(-1, 3)
                        scale = np.max(np.linalg.norm(pts - base, axis=1))
                        if scale < 0.001: scale = 1.0
                        break
                if base is not None:
                    for i in range(SEQUENCE_LENGTH):
                        if np.any(window[i]):
                            window[i, 0::3] = (window[i, 0::3] - base[0]) / scale
                            window[i, 1::3] = (window[i, 1::3] - base[1]) / scale
                            window[i, 2::3] = (window[i, 2::3] - base[2]) / scale

                window = window[np.newaxis, ..., np.newaxis]  # (1, 30, 63, 1)

                probs = model.predict(window, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                raw_label = gesture_labels[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_text = raw_label  # تسمية إنكليزية (raw)

            # تحويل التسمية الإنكليزية لنص عربي للعرض والنطق (لو موجودة بالقاموس)
            display_text = translations.get(detected_text, detected_text) if detected_text else ""
            speech_lang = "ar" if detected_text in translations else "en"

            if detected_text and detected_text != last_spoken:
                speak_async(display_text, lang=speech_lang)
                last_spoken = detected_text
            elif not detected_text:
                last_spoken = ""  # يسمح بإعادة نطق نفس الإشارة بعد ما تختفي اليد

            # --- عرض المعلومات على الشاشة ---
            if buf_size < SEQUENCE_LENGTH:
                # مرحلة التجميع: اعرض شريط التقدم
                overlay = f"جمع بيانات: {buf_size}/{SEQUENCE_LENGTH}"
                frame = draw_text_on_frame(frame, overlay, position=(30, 20))
            elif display_text:
                # تم اكتشاف إشارة بثقة كافية
                overlay = f"{display_text} ({confidence*100:.0f}%)"
                frame = draw_text_on_frame(frame, overlay, position=(30, 20))
            elif raw_label:
                # إشارة محتملة لكن الثقة منخفضة
                overlay = f"? {raw_label} ({confidence*100:.0f}%) - ثقة منخفضة"
                frame = draw_text_on_frame(frame, overlay, position=(30, 20))
            else:
                frame = draw_text_on_frame(frame, "لا توجد يد...", position=(30, 20))
            cv2.imshow("ArSL Real-Time Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
