"""
واجهة المستخدم الرسومية - مترجم لغة الإشارة
=============================================
نفس تصميم الصورة المرفقة بدون زر المحادثة والإعدادات.
"""

import sys
import json
import os
import threading
from collections import deque

import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
import tensorflow as tf

# ── TTS ──────────────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    from playsound3 import playsound
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except ImportError:
    ARABIC_OK = False

# ── إعدادات النموذج ───────────────────────────────────────────────────────────
MODEL_PATH        = "arsl_model.h5"
LABELS_PATH       = "labels.json"
TRANSLATIONS_PATH = "translations.json"
SEQUENCE_LENGTH   = 30
FEATURES_PER_FRAME = 63
CONFIDENCE_THRESHOLD = 0.40   # أخفضنا العتبة لكشف أسرع
VOTING_WINDOW     = 10        # عدد التوقعات المستخدمة في التصويت

ARABIC_FONT_PATH = "C:/Windows/Fonts/arial.ttf"


# ── مساعدات ──────────────────────────────────────────────────────────────────
def load_translations():
    if os.path.exists(TRANSLATIONS_PATH):
        with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def shape_arabic(text: str) -> str:
    if ARABIC_OK:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass
    return text


def speak_async(text: str, lang: str = "ar"):
    """تشغيل الصوت في خيط منفصل - ملف مؤقت فريد لكل استدعاء لتجنب Permission denied."""
    import tempfile, uuid

    def _run():
        # محاولة 1: gTTS (أونلاين - جودة عربية أفضل)
        if GTTS_AVAILABLE:
            try:
                # ملف مؤقت بمسار النظام وبإسم عشوائي لتجنب التضارب
                tmp = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
                gTTS(text=text, lang=lang).save(tmp)
                playsound(tmp)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return
            except Exception as e:
                print(f"[gTTS] {e}")

        # محاولة 2: pyttsx3 (أوفلاين) - engine جديد لكل استدعاء
        try:
            import pyttsx3 as _p
            eng = _p.init()
            eng.setProperty("rate", 150)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except Exception as e:
            print(f"[pyttsx3] {e}")

    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
class SignTranslatorApp:
    # ألوان التصميم
    BG        = "#0d0d0d"
    PANEL     = "#1a1a1a"
    BTN_DARK  = "#1e1e1e"
    BTN_CAM   = "#b8d4e8"      # الأزرق الفاتح لزر الكاميرا
    FG        = "#e8e8e8"
    FG_DIM    = "#888888"
    BORDER    = "#2a2a2a"
    GREEN     = "#4ade80"
    YELLOW    = "#facc15"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("مترجم لغة الإشارة")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        # حجم النافذة
        WIN_W, WIN_H = 560, 820
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - WIN_W) // 2
        y  = (sh - WIN_H) // 2
        self.root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

        # متغيرات الحالة
        self.running      = False
        self.cap          = None
        self.frame_buf    = deque(maxlen=SEQUENCE_LENGTH)
        self.pred_history = deque(maxlen=VOTING_WINDOW)  # سجل التوقعات للتصويت
        self.last_spoken  = ""
        self.model        = None
        self.labels       = []
        self.trans        = {}

        self._load_model()
        self._build_ui()

    # ── تحميل النموذج ─────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return
        self.model  = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.trans = load_translations()

    # ── بناء الواجهة ──────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── العنوان ──
        title_frame = tk.Frame(root, bg=self.BG)
        title_frame.pack(fill="x", pady=(24, 8))

        tk.Label(
            title_frame,
            text="ترجمة الإشارة",
            bg=self.BG,
            fg=self.FG,
            font=("Arial", 22, "bold"),
        ).pack()

        # ── منطقة الكاميرا ──
        cam_outer = tk.Frame(root, bg=self.BORDER, bd=0)
        cam_outer.pack(padx=20, pady=(8, 0))

        self.cam_frame = tk.Frame(cam_outer, bg="#111111", width=520, height=440)
        self.cam_frame.pack_propagate(False)
        self.cam_frame.pack()

        self.cam_label = tk.Label(self.cam_frame, bg="#111111")
        self.cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # نص placeholder
        self.placeholder = tk.Label(
            self.cam_frame,
            text="إدخل إشارة...",
            bg="#111111",
            fg="#555555",
            font=("Arial", 16),
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # ── شريط الإشارة المكتشفة ──
        result_frame = tk.Frame(root, bg=self.BG)
        result_frame.pack(fill="x", padx=20, pady=(12, 0))

        self.result_label = tk.Label(
            result_frame,
            text="",
            bg=self.BG,
            fg=self.GREEN,
            font=("Arial", 18, "bold"),
            anchor="center",
        )
        self.result_label.pack(fill="x")

        self.conf_label = tk.Label(
            result_frame,
            text="",
            bg=self.BG,
            fg=self.FG_DIM,
            font=("Arial", 11),
            anchor="center",
        )
        self.conf_label.pack(fill="x")

        # ── صف الأزرار الوسطى (لغة الإشارة ← العربية) ──
        mid_frame = tk.Frame(root, bg=self.BG)
        mid_frame.pack(fill="x", padx=20, pady=(18, 0))

        btn_style = dict(
            bg=self.BTN_DARK, fg=self.FG,
            font=("Arial", 12),
            relief="flat", bd=0,
            padx=18, pady=12,
            cursor="hand2",
        )

        btn_sign = tk.Button(mid_frame, text="لغة الإشارة", **btn_style)
        btn_sign.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._hover(btn_sign, "#2a2a2a", self.BTN_DARK)

        tk.Label(
            mid_frame, text="→",
            bg=self.BG, fg=self.FG_DIM,
            font=("Arial", 14),
        ).pack(side="left", padx=6)

        btn_ar = tk.Button(mid_frame, text="العربية", **btn_style)
        btn_ar.pack(side="right", expand=True, fill="x", padx=(4, 0))
        self._hover(btn_ar, "#2a2a2a", self.BTN_DARK)

        # ── شريط التقدم (buffer) ──
        self.progress_label = tk.Label(
            root,
            text="",
            bg=self.BG,
            fg=self.FG_DIM,
            font=("Arial", 10),
        )
        self.progress_label.pack(pady=(8, 0))

        # ── زر الكاميرا ──
        bottom_frame = tk.Frame(root, bg=self.BG)
        bottom_frame.pack(fill="x", padx=20, pady=(16, 20))

        self.cam_btn = tk.Button(
            bottom_frame,
            text="كاميرا",
            bg=self.BTN_CAM,
            fg="#1a1a2e",
            font=("Arial", 14, "bold"),
            relief="flat", bd=0,
            padx=40, pady=14,
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.cam_btn.pack(expand=True)
        self._hover(self.cam_btn, "#9dc4dc", self.BTN_CAM)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── تأثير hover ──────────────────────────────────────────────────────────
    def _hover(self, widget, hover_color, normal_color):
        widget.bind("<Enter>", lambda _: widget.config(bg=hover_color))
        widget.bind("<Leave>", lambda _: widget.config(bg=normal_color))

    # ── تشغيل / إيقاف الكاميرا ───────────────────────────────────────────────
    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.config(text="❌ تعذّر فتح الكاميرا", fg="#f87171")
            return
        self.running = True
        self.cam_btn.config(text="إيقاف", bg="#f87171", fg="white")
        self._hover(self.cam_btn, "#e55", "#f87171")
        self.placeholder.place_forget()
        self.frame_buf.clear()
        self.last_spoken = ""
        self._update_frame()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_label.config(image="")
        self.cam_label.image = None
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.result_label.config(text="", fg=self.GREEN)
        self.conf_label.config(text="")
        self.progress_label.config(text="")
        self.cam_btn.config(text="كاميرا", bg=self.BTN_CAM, fg="#1a1a2e")
        self._hover(self.cam_btn, "#9dc4dc", self.BTN_CAM)

    # ── حلقة الفيديو ─────────────────────────────────────────────────────────
    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._stop_camera()
            return

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self.hands.process(rgb)

        # رسم نقاط اليد
        if res.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, res.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS,
            )

        # استخراج landmarks
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                           dtype=np.float32).flatten()
        else:
            vec = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

        self.frame_buf.append(vec)
        buf_size = len(self.frame_buf)

        detected_text = ""
        raw_label     = ""
        confidence    = 0.0
        voted_label   = ""

        if self.model and buf_size == SEQUENCE_LENGTH:
            window = np.array(self.frame_buf, dtype=np.float32)

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

            window = window[np.newaxis, ..., np.newaxis]
            probs      = self.model.predict(window, verbose=0)[0]
            best_idx   = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            raw_label  = self.labels[best_idx]

            # أضف هذا التوقع لسجل التصويت فقط إذا فوق العتبة
            if confidence >= CONFIDENCE_THRESHOLD:
                self.pred_history.append(raw_label)
            else:
                self.pred_history.append("")  # صوت فارغ

            # التصويت: اختر الإشارة الأكثر تكراراً في السجل
            if self.pred_history:
                non_empty = [p for p in self.pred_history if p]
                if non_empty:
                    from collections import Counter
                    most_common, count = Counter(non_empty).most_common(1)[0]
                    # تأكيد الإشارة إذا فازت بأكثر من 40% من الأصوات
                    if count >= max(2, len(self.pred_history) * 0.4):
                        voted_label = most_common

            detected_text = voted_label

        # تحديث نص النتيجة
        if buf_size < SEQUENCE_LENGTH:
            self.progress_label.config(
                text=f"جمع بيانات: {buf_size}/{SEQUENCE_LENGTH}"
            )
            self.result_label.config(text="")
            self.conf_label.config(text="")
        elif detected_text:
            display = self.trans.get(detected_text, detected_text)
            self.result_label.config(text=display, fg=self.GREEN)
            self.conf_label.config(
                text=f"الثقة: {confidence*100:.0f}%", fg=self.FG_DIM
            )
            self.progress_label.config(text="")

            if detected_text != self.last_spoken:
                lang = "ar" if detected_text in self.trans else "en"
                speak_async(display, lang=lang)
                self.last_spoken = detected_text
        elif raw_label and confidence >= CONFIDENCE_THRESHOLD:
            # إشارة محتملة لكن التصويت لم يُثبّتها بعد
            display = self.trans.get(raw_label, raw_label)
            self.result_label.config(text=f"؟ {display}", fg=self.YELLOW)
            self.conf_label.config(
                text=f"ثقة: {confidence*100:.0f}% - انتظر...", fg=self.FG_DIM
            )
            self.progress_label.config(text="")
        else:
            if not res.multi_hand_landmarks:
                self.result_label.config(text="لا توجد يد...", fg=self.FG_DIM)
            else:
                self.result_label.config(text="أثبّت الإشارة...", fg=self.FG_DIM)
            self.conf_label.config(text="")
            self.progress_label.config(text="")
            self.last_spoken = ""

        # عرض الفيديو
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize((520, 440))
        imgtk     = ImageTk.PhotoImage(image=img)
        self.cam_label.config(image=imgtk)
        self.cam_label.image = imgtk

        self.root.after(15, self._update_frame)

    # ── إغلاق ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.hands.close()
        self.root.destroy()


# ── تشغيل ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    root = tk.Tk()
    app  = SignTranslatorApp(root)
    root.mainloop()
