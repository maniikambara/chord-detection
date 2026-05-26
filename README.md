# Chord Detection

Proyek deep learning untuk mengklasifikasikan chord musik yang dimainkan pada piano. Sistem ini menggunakan convolutional neural network (CNN) yang dilatih pada tiga representasi fitur audio: Mel spectrogram, Mel-frequency cepstral coefficients (MFCC), dan Chroma STFT. Model yang telah dilatih dapat mengklasifikasikan 48 jenis chord dan dapat diakses melalui antarmuka web Streamlit untuk prediksi chord tunggal.

## Gambaran Umum

Proyek ini mendemonstrasikan klasifikasi audio end-to-end menggunakan PyTorch. Sistem mengekstrak berbagai representasi fitur dari audio mentah, melatih model CNN terpisah untuk setiap representasi, dan menyediakan antarmuka terpadu untuk prediksi chord secara real-time. Arsitektur yang fleksibel memungkinkan perbandingan berbagai pendekatan rekayasa fitur untuk tugas klasifikasi yang sama.

## Skema Klasifikasi Chord

Sistem mengklasifikasikan chord ke dalam 48 kelas berdasarkan dua atribut: nada dan kualitas chord. Setiap chord dikodekan menggunakan format `{nada}{aksidental}_{tipe}`:

- Nada (7 pilihan): A, B, C, D, E, F, G
- Aksidental (2 pilihan): f (flat) atau n (natural), merepresentasikan 12 nada kromatik
- Tipe chord (4 pilihan): a (augmented), d (diminished), j (mayor), n (minor)

Hal ini menghasilkan 12 nada kromatik x 4 kualitas chord = 48 kelas total.

Contoh: Label `Af_d` merepresentasikan chord A-flat diminished. Dalam nama file audio, label ini mungkin muncul sebagai `piano_3_Af_d_m_45.wav`, di mana komponen tambahan menunjukkan piano, oktaf 3, level dinamik m (mezzaforte), dan nomor contoh 45.

## Struktur Proyek

Proyek diorganisasi sebagai berikut:

| Komponen | Fungsi |
|---|---|
| `cnn.py` | Arsitektur model CNN, loop pelatihan, metrik evaluasi, dan fungsi visualisasi |
| `load.py` | Utilitas pemuatan data fitur, caching, dan encoding label chord |
| `app.py` | Aplikasi web Streamlit untuk prediksi chord tunggal secara interaktif |
| `feature_extraction.ipynb` | Notebook Jupyter untuk mengekstrak fitur audio dari file WAV mentah dan membangun cache fitur |
| `mel_experiment.ipynb` | Pipeline pelatihan dan evaluasi untuk fitur Mel spectrogram |
| `mfcc_experiment.ipynb` | Pipeline pelatihan dan evaluasi untuk fitur MFCC |
| `chroma_experiment.ipynb` | Pipeline pelatihan dan evaluasi untuk fitur Chroma STFT |
| `best_mel_model.pth` | Bobot model pre-trained untuk Mel spectrogram |
| `best_mfcc_model.pth` | Bobot model pre-trained untuk MFCC |
| `best_chroma_model.pth` | Bobot model pre-trained untuk Chroma STFT |

## Instalasi

### Langkah 1: Buat Virtual Environment Python 3.10

Arahkan ke direktori proyek dan buat environment Python yang terisolasi.

**Di Windows:**
```bash
cd chord-detection
python3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Di Linux, macOS, atau WSL:**
```bash
cd chord-detection
python3.10 -m venv venv
source venv/bin/activate
```

### Langkah 2: Instal PyTorch dengan Dukungan GPU (Opsional)

PyTorch sebaiknya diinstal dengan dukungan CUDA 12.4 untuk mengaktifkan akselerasi GPU. Jika tidak memiliki GPU NVIDIA yang kompatibel, hilangkan flag `--index-url` untuk menginstal PyTorch versi CPU saja.

**Dengan dukungan GPU (CUDA 12.4):**
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Instalasi CPU saja:**
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
```

### Langkah 3: Instal Dependensi Proyek

Instal semua paket yang diperlukan yang tercantum dalam requirements.txt:

```bash
pip install -r requirements.txt
```

### Langkah 4: Verifikasi Ketersediaan GPU (Opsional)

Setelah instalasi, verifikasi bahwa PyTorch dapat mengakses GPU:

```bash
python -c "import torch; avail = torch.cuda.is_available(); gpu = torch.cuda.get_device_name(0) if avail else 'None'; print('GPU available:', avail); print('GPU:', gpu)"
```

Output yang diharapkan (jika GPU tersedia):
```
GPU available: True
GPU: NVIDIA RTX 4050
```

Jika GPU tidak terdeteksi, pastikan driver NVIDIA dan CUDA 12.4 sudah terinstal dengan benar dan kompatibel dengan GPU Anda. Untuk memecahkan masalah, instal ulang PyTorch:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Menggunakan Aplikasi Prediksi

Aplikasi Streamlit menyediakan antarmuka yang mudah digunakan untuk mengklasifikasikan chord secara individual.

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser web default pada `http://localhost:8501`.

### Format Input yang Didukung

Aplikasi menerima dua jenis input:

1. **File audio**: File WAV atau MP3 yang mengandung satu chord. Aplikasi akan secara otomatis mengekstrak representasi fitur yang dipilih dari audio mentah.

2. **Tensor fitur yang telah dihitung**: File NumPy .npy yang mengandung array fitur. Bentuk yang diterima adalah:
   - (H, W): Array fitur 2D
   - (H, W, 1): Fitur 2D dengan dimensi channel
   - (1, H, W, 1): Dimensi batch dan channel

### Cara Menggunakan Aplikasi

1. Pilih tipe fitur (mel, mfcc, atau chroma) dari menu dropdown. Tipe fitur ini harus sesuai dengan model yang ingin digunakan.
2. Unggah file audio atau tensor fitur yang telah dihitung sebelumnya.
3. Klik "Deteksi Chord" untuk mengklasifikasikan chord.

Aplikasi menampilkan 5 chord teratas yang diprediksi beserta skor kepercayaannya dan menampilkan visualisasi fitur yang diekstrak.

### Catatan Penting

- Tipe fitur yang dipilih harus kompatibel dengan file yang diunggah. Untuk file audio, aplikasi akan mengekstrak fitur yang sesuai. Untuk file .npy, pastikan bentuk data sesuai dengan dimensi yang diharapkan untuk tipe fitur yang dipilih.
- Dekoding MP3 memerlukan `ffmpeg` atau library `audioread` yang terinstal di sistem.
- Semua audio secara otomatis di-resample ke 16 kHz dan diisi atau dipotong hingga durasi 4 detik sesuai kondisi pelatihan.

## Melatih Model dari Awal

Proyek ini mencakup tiga notebook eksperimen, satu untuk setiap tipe fitur audio. Setiap notebook mengimplementasikan pipeline lengkap: pemuatan data, pelatihan model, evaluasi, dan visualisasi. Jika fitur sudah diekstrak sebelumnya, model dapat dilatih ulang.

### Langkah 1: Ekstrak Fitur dari Audio

Mulai dengan menjalankan notebook ekstraksi fitur, yang memproses file WAV mentah dari dataset pelatihan dan membangun cache fitur:

```
feature_extraction.ipynb
```

Notebook ini membuat tiga direktori cache dengan array NumPy:

```
mel_cache/
  train_x.npy   train_y.npy   val_x.npy   val_y.npy   test_x.npy   test_y.npy
mfcc_cache/
  train_x.npy   train_y.npy   val_x.npy   val_y.npy   test_x.npy   test_y.npy
chroma_cache/
  train_x.npy   train_y.npy   val_x.npy   val_y.npy   test_x.npy   test_y.npy
```

### Langkah 2: Jalankan Notebook Eksperimen

Setelah ekstraksi fitur, buka dan jalankan notebook yang sesuai dengan tipe fitur yang ingin dilatih:

| Notebook | Fitur |
|---|---|
| `mel_experiment.ipynb` | Mel spectrogram (128 bin) |
| `mfcc_experiment.ipynb` | MFCC (13 koefisien) |
| `chroma_experiment.ipynb` | Chroma STFT (12 kelas nada) |

Setiap notebook memuat fitur yang di-cache, memanggil fungsi `run_pipeline` untuk melatih model dengan early stopping, mengevaluasinya pada split pengujian, dan menghasilkan output berikut:

- `best_{fitur}_model.pth`: Bobot model yang disimpan pada epoch dengan validation loss terendah
- `images/{fitur}_training_history.png`: Plot loss dan akurasi pelatihan serta validasi per epoch
- `images/{fitur}_confusion_matrix.png`: Confusion matrix 48x48 yang menampilkan akurasi per kelas

## Detail Teknis

### Arsitektur Model

Arsitektur CNN terdiri dari empat blok konvolusional diikuti oleh kepala classifier:

- **Feature extractor**: Empat blok Conv2d -> BatchNorm -> ReLU -> MaxPool dengan jumlah filter yang meningkat (32 -> 64 -> 128 -> 256). Layer dropout memberikan regularisasi dengan tingkat yang meningkat (0,10 -> 0,30).

- **Adaptive pooling**: Setelah blok konv keempat, AdaptiveAvgPool2d mereduksi peta fitur ke ukuran tetap (4x4) terlepas dari dimensi input.

- **Classifier**: Dua fully-connected layer (4096 -> 512 -> 256 -> 48) dengan aktivasi ReLU dan dropout (0,40 dan 0,30).

Model menggunakan CrossEntropyLoss dengan label smoothing (0,1) dan dilatih dengan optimizer AdamW (lr=0,001, weight_decay=1e-4). Scheduler ReduceLROnPlateau menyesuaikan learning rate selama pelatihan, dan early stopping menghentikan pelatihan jika validation loss tidak membaik setelah ambang batas patience tertentu.

### Spesifikasi Fitur Input

Semua model mengharapkan audio yang diproses pada sample rate 16 kHz dengan durasi 4 detik. Fitur distandarisasi menggunakan normalisasi z-score. Bentuk tensor selama pelatihan adalah N x 1 x H x W, di mana N adalah ukuran batch, H adalah dimensi frekuensi, dan W adalah dimensi waktu.

| Fitur | H (Bin Frekuensi) | W (Frame Waktu) |
|---|---|---|
| Mel spectrogram | 128 | 125 |
| MFCC | 13 | 125 |
| Chroma STFT | 12 | 125 |

Dimensi ini berasal dari pengaturan default librosa: n_fft=2048, hop_length=512, yang menghasilkan 125 frame waktu untuk audio 4 detik pada 16 kHz.

### Konsistensi Encoding Label

Sebuah keputusan desain penting memastikan bahwa indeks kelas chord selalu konsisten antara pelatihan dan inferensi. `LabelEncoder` dalam `load.py` di-fit pada daftar lengkap 48 kelas, bukan hanya pada kelas yang muncul dalam data pelatihan. Hal ini menjamin bahwa label chord tertentu selalu dipetakan ke indeks integer yang sama terlepas dari kelas mana yang muncul dalam split data tertentu. Model selalu memiliki 48 neuron output, satu untuk setiap kelas chord.
