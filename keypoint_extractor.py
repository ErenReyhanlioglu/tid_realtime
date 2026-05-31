import numpy as np
import mediapipe as mp

POSE_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_DIM = len(POSE_INDICES) * 4   # x, y, z, visibility = 44
HAND_DIM = 21 * 3                   # x, y, z = 63
FEATURE_DIM = POSE_DIM + HAND_DIM + HAND_DIM  # 170


class KeypointExtractor:
    def __init__(self):
        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def extract(self, bgr_frame):
        """Return (feature_vector: np.ndarray[170], results) tuple."""
        rgb = bgr_frame[:, :, ::-1]
        results = self._holistic.process(rgb)

        pose = self._pose_features(results)
        left = self._hand_features(results.left_hand_landmarks)
        right = self._hand_features(results.right_hand_landmarks)

        features = np.concatenate([pose, left, right], dtype=np.float32)
        assert features.shape == (FEATURE_DIM,), f"Unexpected shape: {features.shape}"
        return features, results

    def _pose_features(self, results):
        if results.pose_landmarks is None:
            return np.zeros(POSE_DIM, dtype=np.float32)
        lms = results.pose_landmarks.landmark
        out = []
        for idx in POSE_INDICES:
            lm = lms[idx]
            out.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(out, dtype=np.float32)

    def _hand_features(self, hand_landmarks):
        if hand_landmarks is None:
            return np.zeros(HAND_DIM, dtype=np.float32)
        out = []
        for lm in hand_landmarks.landmark:
            out.extend([lm.x, lm.y, lm.z])
        return np.array(out, dtype=np.float32)

    def draw_landmarks(self, bgr_frame, results):
        """Draw pose + hand landmarks onto the frame in-place."""
        dutils = self.mp_draw
        dspec = self.mp_draw.DrawingSpec

        if results.pose_landmarks:
            dutils.draw_landmarks(
                bgr_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                dspec(color=(0, 255, 255), thickness=2, circle_radius=3),
                dspec(color=(0, 180, 180), thickness=2),
            )
        if results.left_hand_landmarks:
            dutils.draw_landmarks(
                bgr_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                dspec(color=(255, 128, 0), thickness=2, circle_radius=3),
                dspec(color=(200, 100, 0), thickness=2),
            )
        if results.right_hand_landmarks:
            dutils.draw_landmarks(
                bgr_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                dspec(color=(0, 128, 255), thickness=2, circle_radius=3),
                dspec(color=(0, 100, 200), thickness=2),
            )

    def close(self):
        self._holistic.close()