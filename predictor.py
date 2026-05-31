import threading
from collections import Counter, deque

import numpy as np


def _build_masked_attention():
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

    return MaskedAttention


WINDOW_SIZE = 82
STEP = 15
CONFIDENCE_THRESHOLD = 0.5
HISTORY_SIZE = 5

# Must match sorted(os.listdir(...)) from training — lexicographic, not numeric order.
CLASSES = [
    "0_ben", "10_var", "11_yok", "12_iyi", "13_kotu",
    "14_nasil", "15_neden", "16_nerede", "17_yardim", "18_doktor",
    "19_hasta", "1_sen", "20_hastane", "21_ilac", "22_gecmis_olsun",
    "23_ataturk", "24_ev", "25_zaman", "26_icmek", "27_yemek",
    "28_yapmak", "29_bakmak", "2_selam", "3_hoscakal", "4_tamam",
    "5_evet", "6_hayir", "7_tesekkur", "8_rica_etmek", "9_ozur_dilemek",
]


class Predictor:
    def __init__(self, model_path: str, norm_path: str):
        import tensorflow as tf  # deferred import — TF is slow to load

        MaskedAttention = _build_masked_attention()
        self._model = tf.keras.models.load_model(
            model_path,
            custom_objects={"MaskedAttention": MaskedAttention},
            compile=False,
        )
        # Warm up — first call is always slow
        dummy = np.zeros((1, WINDOW_SIZE, 170), dtype=np.float32)
        self._model.predict(dummy, verbose=0)

        norm = np.load(norm_path)
        self._mean: np.ndarray = norm["mean"].astype(np.float32)
        self._std: np.ndarray = norm["std"].astype(np.float32)

        self._buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self._history: deque[str] = deque(maxlen=HISTORY_SIZE)
        self._frame_counter = 0

        # Inference runs in a background thread; _inferring prevents overlap.
        self._inferring = False
        self._lock = threading.Lock()

        self.label = "?"
        self.confidence = 0.0
        self.history: list[str] = []

    def update(self, feature_vector: np.ndarray) -> None:
        self._buffer.append(feature_vector)
        self._frame_counter += 1

        if self._frame_counter % STEP != 0:
            return

        # Skip if the previous inference hasn't finished yet.
        with self._lock:
            if self._inferring:
                return
            self._inferring = True

        window = self._build_window()
        threading.Thread(
            target=self._infer_thread,
            args=(window,),
            daemon=True,
        ).start()

    def buffer_fill(self) -> int:
        return len(self._buffer)

    def _infer_thread(self, window: np.ndarray) -> None:
        try:
            probs = self._model.predict(window, verbose=0)[0]
            idx  = int(np.argmax(probs))
            conf = float(probs[idx])

            with self._lock:
                self._history.append(CLASSES[idx])
                self.history = list(self._history)
                voted = Counter(self._history).most_common(1)[0][0]
                self.label = voted if conf >= CONFIDENCE_THRESHOLD else "?"
                self.confidence = conf
        finally:
            with self._lock:
                self._inferring = False

    def _build_window(self) -> np.ndarray:
        buf = list(self._buffer)
        pad_len = WINDOW_SIZE - len(buf)
        if pad_len > 0:
            pad = np.full((pad_len, buf[0].shape[0]), -1.0, dtype=np.float32)
            buf = buf + list(pad)
        window = np.stack(buf, axis=0)  # (82, 170)

        mask = (window != -1.0).all(axis=-1)  # (82,)
        window = np.where(
            mask[:, np.newaxis],
            (window - self._mean) / self._std,
            -1.0,
        )
        return window[np.newaxis]  # (1, 82, 170)