"""
تدريب نموذج CNN-LSTM (Model Training)
======================================
يطابق قسم 3.2 (Deep Learning in Sign Language Recognition) بالبحث:
    - طبقات Conv1D تلعب دور الـ CNN: تستخرج ميزات مكانية (spatial features)
      من الـ 63 رقم (21 نقطة × x,y,z) بكل فريم على حدة (نفس فكرة Figure 2).
    - طبقات LSTM تلعب دور الجزء الزمني: تتعلم كيف تتغير هالميزات عبر
      الفريمات المتتالية (نفس فكرة Figure 3 / CRNN).

ملاحظة مهمة: البحث يشرح CNN تشتغل على صورة اليد الخام، بينما هذا الكود
يشتغل على إحداثيات MediaPipe (landmarks) لتقليل التعقيد الحسابي —
وهذا بالضبط اللي وصفه القسم 3.3.1 بأن استخراج الإحداثيات بدل الصورة
الكاملة "يقلل التعقيد الحسابي بشكل كبير". يعني الكود يطابق فكرة البحث
بدون تناقض.

الاستخدام:
    python 2_train_model.py
"""

import os
import sys
import json

# Fix UnicodeEncodeError in Windows CMD
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras import layers, models, callbacks

DATASET_DIR = "dataset"
MODEL_OUT = "arsl_model.h5"
LABELS_OUT = "labels.json"

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63  # 21 landmarks * (x, y, z)


def load_dataset():
    sequences, labels = [], []
    gesture_names = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not gesture_names:
        raise RuntimeError(
            "ما فيه بيانات بمجلد dataset/. شغّل 1_collect_data.py أول شي."
        )

    for gesture in gesture_names:
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        for fname in os.listdir(gesture_dir):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(gesture_dir, fname))
                if arr.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    # Normalization: subtract wrist and divide by scale
                    base = None
                    scale = 1.0
                    for i in range(SEQUENCE_LENGTH):
                        if np.any(arr[i]):
                            base = arr[i, 0:3]
                            pts = arr[i].reshape(-1, 3)
                            scale = np.max(np.linalg.norm(pts - base, axis=1))
                            if scale < 0.001: scale = 1.0
                            break
                    if base is not None:
                        for i in range(SEQUENCE_LENGTH):
                            if np.any(arr[i]):
                                arr[i, 0::3] = (arr[i, 0::3] - base[0]) / scale
                                arr[i, 1::3] = (arr[i, 1::3] - base[1]) / scale
                                arr[i, 2::3] = (arr[i, 2::3] - base[2]) / scale
                    sequences.append(arr)
                    labels.append(gesture)

    return np.array(sequences, dtype=np.float32), np.array(labels), gesture_names


def build_cnn_lstm_model(num_classes: int) -> tf.keras.Model:
    """
    البنية الهجينة CNN + LSTM (كما طلبت):
      TimeDistributed(Conv1D) -> استخراج ميزات مكانية (Spatial Features) من النقاط.
      LSTM -> تعلم النمط الزمني (Temporal Patterns) عبر الفريمات.
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME, 1))

    # ----- الجزء الشبيه بـ CNN -----
    x = layers.TimeDistributed(
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")
    )(inputs)
    x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
    x = layers.TimeDistributed(
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")
    )(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)

    # ----- الجزء الزمني: LSTM -----
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y_raw, gesture_names = load_dataset()

    encoder = LabelEncoder()
    encoder.fit(gesture_names)
    y = encoder.transform(y_raw)

    # إضافة بعد القناة (channel) عشان Conv1D
    X = X[..., np.newaxis]  # shape: (N, 30, 63, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn_lstm_model(num_classes=len(gesture_names))
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nدقة التحقق (Validation Accuracy): {val_acc * 100:.2f}%")

    model.save(MODEL_OUT)
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(list(encoder.classes_), f, ensure_ascii=False, indent=2)

    print(f"تم حفظ النموذج في: {MODEL_OUT}")
    print(f"تم حفظ أسماء التصنيفات في: {LABELS_OUT}")


if __name__ == "__main__":
    main()