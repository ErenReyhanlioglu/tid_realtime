# TID Realtime — Türk İşaret Dili Gerçek Zamanlı Tahmin

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Webcam üzerinden 30 TİD kelimesini gerçek zamanlı olarak tanıyan bir yapay zeka uygulaması. MediaPipe ile vücut ve el iskelet noktaları çıkarılır, eğitilmiş bir LSTM + Attention modeli ile kelime tahmini yapılır.

---

## Nasıl çalışır?

```
Webcam → MediaPipe Holistic → 170 boyutlu özellik vektörü
       → 82 framelık kayan pencere → LSTM + Masked Attention modeli
       → Softmax (30 sınıf) → Majority voting → Ekran overlay
```

Her frame'de MediaPipe 11 vücut noktası (pose) + sol el + sağ el keypoint'lerini çıkarır (toplam 170 sayı). Bu vektörler 82 framelık bir pencerede birikir; her 15 frame'de bir model bu pencereyi değerlendirip tahmin üretir. Son 5 tahminden en çok tekrar eden kelime gösterilir.

---

## Desteklenen kelimeler (30 sınıf)

| | | | |
|---|---|---|---|
| ben | sen | selam | hoşçakal |
| tamam | evet | hayır | teşekkür |
| rica etmek | özür dilemek | var | yok |
| iyi | kötü | nasıl | neden |
| nerede | yardım | doktor | hasta |
| hastane | ilaç | geçmiş olsun | atatürk |
| ev | zaman | içmek | yemek |
| yapmak | bakmak | | |

---

## Gereksinimler

- **Python 3.11 veya üzeri**
- **Webcam** (USB veya dahili, index 0)
- Windows, Linux veya macOS

---

## Kurulum

### Seçenek A — `uv` ile (önerilen)

`uv` modern ve hızlı bir Python paket yöneticisidir. Hem Python sürümünü hem sanal ortamı otomatik yönetir.

**1. uv kur (Windows PowerShell — yönetici olarak):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Kurulumdan sonra terminali kapatıp yeniden aç.

**2. Proje klasörüne gel:**

```powershell
cd tid_realtime
```

**3. Bağımlılıkları kur:**

```powershell
uv sync
```

Bu komut Python 3.11 ortamı oluşturur ve tüm paketleri `uv.lock` dosyasına göre tam sürümüyle yükler.

---

### Seçenek B — standart `pip` ile

`uv` kurmak istemiyorsan:

```powershell
pip install tensorflow-cpu mediapipe==0.10.14 opencv-python numpy
```

---

## Çalıştırma

### Gerçek zamanlı tahmin (webcam)

```powershell
uv run python main.py
# veya pip kullanıyorsan:
python main.py
```

Çıkmak için kamera penceresinde **`q`** tuşuna bas.

### Video dosyalarıyla offline değerlendirme

`sign_samples/` klasörüne `0_ben.mp4`, `5_evet.mp4` gibi isimlendirilmiş `.mp4` videoları koy.

**Yöntem 1** — Eğitim pipeline'ıyla birebir aynı değerlendirme (tüm frame → center-crop/pad → normalize → tahmin):

```powershell
uv run python evaluate_samples.py
```

**Yöntem 2** — main.py'nin aynı videoyu işleyeceği tahmini simüle eder (kayan pencere + majority voting):

```powershell
uv run python evaluate_live_sim.py
```

Her iki script de şu formatta çıktı verir:

```
Video                     Gercek               Tahmin               Guven   Sonuc
─────────────────────────────────────────────────────────────────────────────────────
0_ben.mp4                 0_ben                0_ben                 91.2%  ✓
5_evet.mp4                5_evet               5_evet                87.4%  ✓
6_hayir.mp4               6_hayir              6_hayir               78.1%  ✓

Sonuc: 3/3 dogru  (100.0%)
```

---

## Ekran overlay açıklaması

| Bölge | İçerik |
|---|---|
| Sol üst — büyük yeşil yazı | Tahmin edilen kelime |
| Altı | Güven yüzdesi |
| Mavi/yeşil bar | Buffer doluluk oranı (mavi = dolmakta, yeşil = hazır) |
| Sağ üst panel | Son 5 tahmin geçmişi (en yeni üstte) |
| Vücut üzerindeki çizgiler | MediaPipe landmark görselleştirmesi |

Terminal çıktısı (yalnızca tahmin değiştiğinde):

```
[TAHMiN] 2_selam                   Guven:  87.3%  Buffer: 82/82
[TAHMiN] 5_evet                    Guven:  91.0%  Buffer: 82/82
```

---

## Proje yapısı

```
tid_realtime/
├── main.py                  # Webcam döngüsü ve overlay
├── predictor.py             # Kayan pencere, model yükleme, inference thread
├── keypoint_extractor.py    # MediaPipe holistic → 170 boyutlu vektör
├── evaluate_samples.py      # Video dosyaları ile offline değerlendirme
├── evaluate_live_sim.py     # main.py davranışını simüle eden değerlendirme
├── pyproject.toml           # Bağımlılık tanımları
├── uv.lock                  # Kilitlenmiş paket sürümleri
├── models/
│   ├── model_final.keras    # Eğitilmiş LSTM + Attention modeli (17 MB)
│   └── normalization.npz    # Eğitim verisinden hesaplanan mean ve std
└── sign_samples/            # (isteğe bağlı) Test video klipleri
```

---

## Teknik detaylar

| Parametre | Değer |
|---|---|
| Özellik vektörü | 170 boyut — pose 44 + sol el 63 + sağ el 63 |
| Pencere boyutu | 82 frame |
| Tahmin adımı | Her 15 frame'de bir |
| Padding değeri | -1.0 (eksik frame için) |
| Normalizasyon | (x − mean) / std, yalnızca gerçek frame'lerde |
| Majority voting | Son 5 tahmin |
| Güven eşiği | 0.50 (altındaysa "?" gösterilir) |
| Model mimarisi | Bidirectional LSTM + Masked Attention + Dense(30, softmax) |
| Model çıktısı | 30 sınıf softmax olasılıkları |
| Inference | Arka plan thread'i (ana döngüyü bloklamaz) |

---

## Sorun giderme

**`[HATA] Kamera acilamadi (index 0)`**
Başka bir uygulama kamerayı kullanıyor olabilir. Kapat ve tekrar dene. Birden fazla kamera varsa `main.py` içinde `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)` dene.

**Model yükleme çok yavaş**
TensorFlow CPU ilk yüklemede yavaştır (30–60 sn), normaldir. İlk `predict` çağrısı da ısınma için yapılır.

**Tahmin hep `?` çıkıyor**
Güven 0.50'nin altında kalıyor demektir. Yeterli aydınlatma olduğundan ve kameraya dik baktığından emin ol. Buffer barının yeşile dönmesini bekle.

**`mediapipe` veya `tensorflow` import hatası**
`uv sync` veya `pip install` adımını atlamış olabilirsin. Kurulumu tekrar çalıştır.

---

## Veri ve eğitim notebookları

Eğitim verisi, veri çıkarım scriptleri ve Jupyter notebookları Google Drive'da:

[Google Drive — TID Proje Dosyaları](https://drive.google.com/drive/folders/1CWbJuS-4jW6sbJxt4ujwVbuEXoXlzzXo?usp=sharing)

---

## Lisans

[MIT](LICENSE)
