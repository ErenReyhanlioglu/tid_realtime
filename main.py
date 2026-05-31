import os
import sys
import cv2
import numpy as np

from keypoint_extractor import KeypointExtractor
from predictor import Predictor, WINDOW_SIZE

MODEL_PATH = os.path.join("models", "model_final.keras")
NORM_PATH = os.path.join("models", "normalization.npz")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(frame, text, pos, scale, color, thickness=2):
    cv2.putText(frame, text, pos, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, label, confidence, buf_fill, history):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    put_text(frame, label, (10, 50), 1.4, (0, 255, 80), 2)

    conf_str = f"Guven: {confidence * 100:.1f}%"
    put_text(frame, conf_str, (10, 90), 0.7, (255, 255, 255), 2)

    fill_ratio = buf_fill / WINDOW_SIZE
    bar_w = 280
    cv2.rectangle(frame, (10, 108), (10 + bar_w, 126), (60, 60, 60), -1)
    filled_w = int(bar_w * fill_ratio)
    bar_color = (0, 200, 255) if fill_ratio < 1.0 else (0, 255, 100)
    cv2.rectangle(frame, (10, 108), (10 + filled_w, 126), bar_color, -1)
    put_text(frame, f"Buffer: {buf_fill}/{WINDOW_SIZE}", (10, 148), 0.55,
             (200, 200, 200), 1)

    panel_x = w - 210
    overlay2 = frame.copy()
    panel_h = 20 + len(history) * 24 + 10
    cv2.rectangle(overlay2, (panel_x - 8, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.45, frame, 0.55, 0, frame)

    put_text(frame, "Gecmis:", (panel_x, 18), 0.5, (180, 180, 180), 1)
    for i, h_label in enumerate(reversed(history)):
        shade = max(80, 255 - i * 40)
        put_text(frame, h_label, (panel_x, 40 + i * 24), 0.55,
                 (shade, shade, shade), 1)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model bulunamadi: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(NORM_PATH):
        print(f"[HATA] Normalizasyon dosyasi bulunamadi: {NORM_PATH}")
        sys.exit(1)

    print("Model yukleniyor...")
    predictor = Predictor(MODEL_PATH, NORM_PATH)
    extractor = KeypointExtractor()
    print("Model yuklendi. Kamera aciliyor...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera acilamadi (index 0).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_label = None
    print("Basladi. Cikmak icin 'q' tusuna basin.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[UYARI] Frame alinamadi, yeniden deneniyor...")
            continue

        # Training data (AUTSL) was not flipped — keep consistent.
        # Uncomment the line below only if inference looks mirrored or wrong.
        # frame = cv2.flip(frame, 1)

        features, results = extractor.extract(frame)
        predictor.update(features)

        if predictor.label != prev_label:
            prev_label = predictor.label
            print(
                f"[TAHMiN] {predictor.label:25s}  "
                f"Guven: {predictor.confidence * 100:5.1f}%  "
                f"Buffer: {predictor.buffer_fill()}/{WINDOW_SIZE}"
            )

        extractor.draw_landmarks(frame, results)
        draw_overlay(
            frame,
            predictor.label,
            predictor.confidence,
            predictor.buffer_fill(),
            predictor.history,
        )

        cv2.imshow("TID Realtime - Turk Isaret Dili", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("Program kapatildi.")


if __name__ == "__main__":
    main()