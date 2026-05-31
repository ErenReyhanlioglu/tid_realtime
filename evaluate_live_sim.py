"""
Live pipeline simulator — feeds sign_samples videos frame by frame
through the exact same Predictor sliding-window logic used in main.py.
Shows what main.py would predict for each video.
"""

import os
import sys
import cv2

SAMPLES_DIR = "sign_samples"
MODEL_PATH  = os.path.join("models", "model_final.keras")
NORM_PATH   = os.path.join("models", "normalization.npz")


def main():
    for path in [MODEL_PATH, NORM_PATH, SAMPLES_DIR]:
        if not os.path.exists(path):
            print(f"[HATA] Bulunamadi: {path}")
            sys.exit(1)

    print("Model yukleniyor...")
    from predictor import Predictor
    from keypoint_extractor import KeypointExtractor

    extractor = KeypointExtractor()
    predictor = Predictor(MODEL_PATH, NORM_PATH)

    videos = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".mp4")])

    print(f"\n{'Video':<25} {'Gercek':<20} {'Tahmin':<20} {'Guven':>7}  {'Sonuc'}")
    print("─" * 85)

    correct = 0

    for video_file in videos:
        true_label = os.path.splitext(video_file)[0]
        video_path = os.path.join(SAMPLES_DIR, video_file)

        # Reset predictor state for each video
        from collections import deque
        predictor._buffer.clear()
        predictor._history.clear()
        predictor._frame_counter = 0
        predictor.label = "?"
        predictor.confidence = 0.0
        predictor.history = []

        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # No flip — matches training data convention
            features, _ = extractor.extract(frame)
            predictor.update(features)
        cap.release()

        pred = predictor.label
        conf = predictor.confidence
        ok   = pred == true_label
        if ok:
            correct += 1

        mark = "✓" if ok else "✗"
        print(f"{video_file:<25} {true_label:<20} {pred:<20} {conf*100:>6.1f}%  {mark}")

    extractor.close()

    total = len(videos)
    print("─" * 85)
    print(f"\nSonuc: {correct}/{total} dogru  ({correct/total*100:.1f}%)")
    print("\nNot: Bu test, main.py'nin ayni videolari islediginde ne tahmin")
    print("edecegini simule eder (sliding window + majority voting).")


if __name__ == "__main__":
    main()