"""
Offline evaluation on sign_samples/ videos.
Replicates the training pipeline exactly:
  video → extract all frames → center-crop or tail-pad → normalize (mask) → predict
"""

import os
import sys
import cv2
import numpy as np

SAMPLES_DIR = "sign_samples"
MODEL_PATH  = os.path.join("models", "model_final.keras")
NORM_PATH   = os.path.join("models", "normalization.npz")
MAX_SEQ_LEN = 82

# Lexicographic order — matches sorted(os.listdir(...)) from training
CLASSES = [
    "0_ben", "10_var", "11_yok", "12_iyi", "13_kotu",
    "14_nasil", "15_neden", "16_nerede", "17_yardim", "18_doktor",
    "19_hasta", "1_sen", "20_hastane", "21_ilac", "22_gecmis_olsun",
    "23_ataturk", "24_ev", "25_zaman", "26_icmek", "27_yemek",
    "28_yapmak", "29_bakmak", "2_selam", "3_hoscakal", "4_tamam",
    "5_evet", "6_hayir", "7_tesekkur", "8_rica_etmek", "9_ozur_dilemek",
]
POSE_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
NUM_FEATURES = 170


# ── Feature extraction (identical to TID_Data_Eren.ipynb) ──────────────────────

def extract_keypoints(frame_bgr, holistic):
    import mediapipe as mp
    rgb = frame_bgr[:, :, ::-1]
    results = holistic.process(rgb)

    if results.pose_landmarks:
        pose = np.array([
            [results.pose_landmarks.landmark[i].x,
             results.pose_landmarks.landmark[i].y,
             results.pose_landmarks.landmark[i].z,
             results.pose_landmarks.landmark[i].visibility]
            for i in POSE_INDICES
        ]).flatten()
    else:
        pose = np.zeros(len(POSE_INDICES) * 4, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)


def video_to_sequence(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints(frame, holistic))
    cap.release()
    return np.array(frames, dtype=np.float32)  # (T, 170)


# ── Padding (identical to load_split in TID_Model_Eren.ipynb cell-16) ──────────

def pad_or_crop(seq, max_len, num_features):
    T = seq.shape[0]
    if T >= max_len:
        start = (T - max_len) // 2
        return seq[start:start + max_len]
    else:
        pad = np.full((max_len - T, num_features), -1.0, dtype=np.float32)
        return np.vstack([seq, pad])


# ── Normalization (identical to cell-17) ────────────────────────────────────────

def normalize(seq, mean, std):
    mask = (seq != -1.0).all(axis=-1)  # (T,)
    return np.where(mask[:, np.newaxis], (seq - mean) / std, -1.0)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    import mediapipe as mp

    for path in [MODEL_PATH, NORM_PATH, SAMPLES_DIR]:
        if not os.path.exists(path):
            print(f"[HATA] Bulunamadi: {path}")
            sys.exit(1)

    print("Model yukleniyor...")
    import tensorflow as tf
    from tensorflow.keras import layers

    class MaskedAttention(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.score_dense = layers.Dense(1, activation="tanh")
        def call(self, x, mask=None):
            scores = self.score_dense(x)
            scores = tf.squeeze(scores, axis=-1)
            if mask is not None:
                scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9
            weights = tf.nn.softmax(scores, axis=-1)
            weights = tf.expand_dims(weights, axis=-1)
            return tf.reduce_sum(x * weights, axis=1)
        def get_config(self):
            return super().get_config()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"MaskedAttention": MaskedAttention},
        compile=False,
    )

    norm = np.load(NORM_PATH)
    mean = norm["mean"].astype(np.float32)
    std  = norm["std"].astype(np.float32)

    videos = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".mp4")])
    if not videos:
        print("[HATA] sign_samples/ icinde .mp4 bulunamadi.")
        sys.exit(1)

    print(f"\n{'Video':<25} {'Gercek':<20} {'Tahmin':<20} {'Guven':>7}  {'Sonuc'}")
    print("─" * 85)

    correct = 0
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for video_file in videos:
        true_label = os.path.splitext(video_file)[0]  # e.g. "0_ben"
        video_path = os.path.join(SAMPLES_DIR, video_file)

        seq = video_to_sequence(video_path, holistic)

        if len(seq) == 0:
            print(f"{video_file:<25} {true_label:<20} {'HATA: frame yok'}")
            continue

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq = pad_or_crop(seq, MAX_SEQ_LEN, NUM_FEATURES)
        seq = normalize(seq, mean, std)

        inp   = seq[np.newaxis]  # (1, 82, 170)
        probs = model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        pred  = CLASSES[idx]

        ok = pred == true_label
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"

        print(f"{video_file:<25} {true_label:<20} {pred:<20} {conf*100:>6.1f}%  {mark}")

    holistic.close()

    total = len(videos)
    print("─" * 85)
    print(f"\nSonuc: {correct}/{total} dogru  ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    main()