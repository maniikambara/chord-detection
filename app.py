# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import tempfile

os.environ['OMP_NUM_THREADS'] = '1'
if sys.platform == 'win32':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import streamlit as st
import torch

torch.set_num_threads(1)

try:
    import librosa
except ImportError:
    librosa = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

HAS_AUDIO_INPUT = hasattr(st, "audio_input")

from cnn import CNNModel
from load import LABELS, CHORD_TYPE_NAMES

FEATURE_CONFIG = {
    "mel": {
        "weight": Path("best_mel_model.pth"),
        "display_name": "Mel Spectrogram",
        "description": "Representasi frekuensi yang meniru persepsi pendengaran manusia; cocok untuk audio umum.",
    },
    "mfcc": {
        "weight": Path("best_mfcc_model.pth"),
        "display_name": "MFCC",
        "description": "Koefisien cepstral frekuensi-mel yang menangkap karakteristik timbral dan tekstur bunyi.",
    },
    "chroma": {
        "weight": Path("best_chroma_model.pth"),
        "display_name": "Chroma STFT",
        "description": "Representasi kelas nada kromatik; ideal untuk analisis konten harmonis dan tonal.",
    },
}

SR         = 16000
DURATION   = 4
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
N_MFCC     = 13


# -- Audio / feature utilities -------------------------------------------------

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    peak  = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Reduce background noise from microphone recordings for cleaner chord detection.

    Pipeline:
    1. Bandpass filter (80-7500 Hz) to isolate piano-relevant frequencies and
       remove low-frequency rumble, electrical hum, and high-frequency hiss.
    2. Spectral gating: estimate the noise floor from the quietest frames
       and subtract it from the magnitude spectrum.
    3. Trim leading and trailing silence.
    """
    try:
        from scipy.signal import butter, sosfilt
    except ImportError:
        return audio

    # --- Stage 1: Bandpass filter ---
    nyquist = sr / 2
    lo = 80.0 / nyquist
    hi = min(7500.0 / nyquist, 0.99)
    sos = butter(5, [lo, hi], btype="band", output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)

    # --- Stage 2: Spectral gating ---
    n_fft_sg = 2048
    hop_sg = 512
    stft = librosa.stft(audio, n_fft=n_fft_sg, hop_length=hop_sg)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Noise profile: mean magnitude of the quietest 10 % of frames
    frame_energy = np.sum(magnitude ** 2, axis=0)
    n_noise = max(1, int(len(frame_energy) * 0.10))
    noise_idx = np.argsort(frame_energy)[:n_noise]
    noise_profile = np.mean(magnitude[:, noise_idx], axis=1, keepdims=True)

    # Subtract with a safety floor to prevent musical-noise artifacts
    magnitude_clean = np.maximum(magnitude - 1.5 * noise_profile, magnitude * 0.05)
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio = librosa.istft(stft_clean, hop_length=hop_sg, length=len(audio)).astype(np.float32)

    # --- Stage 3: Trim silence ---
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) >= int(sr * 0.3):
        audio = trimmed

    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - len(audio)), mode="constant")


def standardize(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32)
    return (feature - np.mean(feature)) / (np.std(feature) + 1e-8)


def extract_feature_from_audio(
    file_bytes: bytes, feature_type: str, suffix: str = ".wav",
    denoise: bool = False,
) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa belum terpasang. Jalankan `pip install librosa`.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        audio, _ = librosa.load(tmp_path, sr=SR, mono=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    audio = normalize_audio(audio)
    if denoise:
        audio = denoise_audio(audio, SR)
        audio = normalize_audio(audio)
    audio = pad_or_truncate(audio, SR * DURATION)

    if feature_type == "mel":
        feature = librosa.feature.melspectrogram(
            y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        feature = librosa.power_to_db(feature, ref=np.max)
    elif feature_type == "mfcc":
        feature = librosa.feature.mfcc(
            y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
    elif feature_type == "chroma":
        feature = librosa.feature.chroma_stft(
            y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
    else:
        raise ValueError(f"Tipe fitur tidak didukung: {feature_type}")

    return standardize(feature)


@st.cache_resource
def load_model(feature_type: str):
    weight_path = FEATURE_CONFIG[feature_type]["weight"]
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Bobot model tidak ditemukan: {weight_path}. "
            "Latih model terlebih dahulu melalui notebook eksperimen yang sesuai."
        )
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = CNNModel(num_classes=len(LABELS))
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def predict(feature: np.ndarray, model, device) -> np.ndarray:
    tensor = torch.from_numpy(feature).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probabilities


def format_label(label: str) -> str:
    note_raw, type_code = label.split("_")
    note_letter = note_raw[0]
    accidental  = "\u266d" if note_raw[1] == "f" else ""  # flat sign
    chord_type  = CHORD_TYPE_NAMES.get(type_code, type_code)
    return f"{note_letter}{accidental} {chord_type}"


def render_feature_map(feature: np.ndarray, feature_key: str) -> None:
    sq = feature.squeeze()
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor("#0f1628")
        ax.set_facecolor("#0f1628")
        cmap = "magma" if feature_key in ("mel", "mfcc") else "viridis"
        im   = ax.imshow(sq, aspect="auto", origin="lower", cmap=cmap, interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
        cbar.ax.tick_params(colors="#6b7db3", labelsize=7)
        cbar.outline.set_edgecolor("#232b4a")
        ax.set_xlabel("Time frames", color="#6b7db3", fontsize=8, labelpad=6)
        ax.set_ylabel("Freq bins",   color="#6b7db3", fontsize=8, labelpad=6)
        ax.tick_params(colors="#6b7db3", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#232b4a")
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        mn, mx   = sq.min(), sq.max()
        img_norm = (sq - mn) / (mx - mn + 1e-8)
        st.image(img_norm, use_container_width=True)


# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="ChordNet",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Session state -------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# -- Global CSS ----------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg:           #080c1f;
    --surface:      #101529;
    --surface-alt:  #141b33;
    --surface-hi:   #192040;
    --border:       #1e2847;
    --border-light: #283560;
    --accent:       #5b5ef4;
    --accent-mid:   #6366f1;
    --accent-light: #818cf8;
    --accent-soft:  rgba(91,94,244,0.14);
    --accent-glow:  rgba(91,94,244,0.22);
    --success:      #34d399;
    --warning:      #fbbf24;
    --text:         #e8edff;
    --text-sec:     #b8c4e8;
    --text-muted:   #5a6a96;
    --mono:         'JetBrains Mono', monospace;
    --sans:         'DM Sans', sans-serif;
    --r-sm:         8px;
    --r-md:         12px;
    --r-lg:         18px;
    --r-xl:         24px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(91,94,244,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(91,94,244,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Page header */
.page-header {
    padding: 2.5rem;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.page-header::after {
    content: '';
    position: absolute;
    top: -60px;
    right: -60px;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(91,94,244,0.1) 0%, transparent 70%);
    pointer-events: none;
}

.page-header::before {
    content: '';
    position: absolute;
    bottom: -40px;
    left: 40%;
    width: 360px;
    height: 160px;
    background: radial-gradient(ellipse, rgba(91,94,244,0.05) 0%, transparent 70%);
    pointer-events: none;
}

.page-title {
    font-size: clamp(2.2rem, 4.5vw, 3.4rem);
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.035em;
    line-height: 1.05;
    margin-bottom: 1rem;
}

.page-title .grad {
    background: linear-gradient(120deg, #818cf8 10%, #c4b5fd 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    font-size: 0.92rem;
    font-weight: 400;
    color: var(--text-muted);
    max-width: 500px;
    line-height: 1.8;
    margin: 0;
}

/* Column layout */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    border-right: 1px solid var(--border);
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    padding: 1.75rem 1.5rem !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    padding: 1.75rem 2.5rem !important;
}

/* Field label */
.field-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}

/* Horizontal divider */
.hdivider {
    height: 1px;
    background: var(--border);
    margin: 1.1rem 0;
}

/* File uploader */
[data-testid="stFileUploader"] > div:first-child {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border-light) !important;
    border-radius: var(--r-md) !important;
    padding: 1.2rem 1rem !important;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
}

[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: var(--accent-mid) !important;
    background: var(--accent-soft) !important;
    box-shadow: 0 0 0 3px rgba(91,94,244,0.1) !important;
}

[data-testid="stFileDropzoneInstructions"] span,
[data-testid="stFileDropzoneInstructions"] small {
    color: var(--text-muted) !important;
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
}

/* Audio player */
audio {
    width: 100%;
    height: 36px;
    border-radius: var(--r-sm);
    margin-top: 0.65rem;
    accent-color: var(--accent-mid);
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    gap: 0.45rem !important;
    flex-direction: column !important;
}

[data-testid="stRadio"] label {
    background: var(--surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--r-md) !important;
    padding: 0.65rem 1rem !important;
    cursor: pointer;
    transition: border-color 0.18s, background 0.18s, box-shadow 0.18s;
    margin: 0 !important;
    width: 100% !important;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(91,94,244,0.55) !important;
    background: rgba(91,94,244,0.07) !important;
}

[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
    border-color: var(--accent-mid) !important;
    background: var(--accent-soft) !important;
    box-shadow: 0 0 0 3px rgba(91,94,244,0.1), inset 0 0 0 1px rgba(91,94,244,0.2) !important;
}

[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    margin: 0;
}

/* Description box */
.desc-box {
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.7;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.6rem 0.85rem;
    margin-top: 0.5rem;
}

/* Detect button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-mid) 50%, #7779fa 100%) !important;
    color: #fff !important;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    padding: 0.75rem 0 !important;
    height: auto !important;
    width: 100% !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 24px rgba(91,94,244,0.4), 0 1px 4px rgba(0,0,0,0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(91,94,244,0.5), 0 2px 8px rgba(0,0,0,0.4) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(91,94,244,0.35) !important;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 500px;
    gap: 0.75rem;
    text-align: center;
    padding: 2rem;
}

.empty-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-sec);
    letter-spacing: -0.015em;
}

.empty-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    max-width: 300px;
    line-height: 1.8;
    margin-bottom: 0.5rem;
}

.step-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    text-align: left;
    width: 100%;
    max-width: 280px;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.55rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
}

.step-num {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent-light);
    background: var(--accent-soft);
    border: 1px solid rgba(91,94,244,0.35);
    width: 20px;
    height: 20px;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    line-height: 1;
}

.step-text {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
    padding-top: 1px;
}

/* Chord result card */
.chord-display {
    background: linear-gradient(145deg, var(--surface) 0%, var(--surface-alt) 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--r-xl);
    padding: 2rem 2.2rem 1.75rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), 0 0 0 1px rgba(91,94,244,0.08);
}

.chord-display::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 50%, transparent 100%);
}

.chord-display::after {
    content: '';
    position: absolute;
    top: -30px;
    right: -30px;
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, rgba(91,94,244,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.chord-meta {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-light);
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

.chord-name {
    font-family: var(--mono);
    font-size: clamp(2.6rem, 5vw, 4rem);
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    z-index: 1;
}

.conf-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    white-space: nowrap;
    min-width: 78px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.conf-track {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: visible;
    position: relative;
}

.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%);
    box-shadow: 0 0 10px rgba(91,94,244,0.6);
    position: relative;
}

.conf-fill::after {
    content: '';
    position: absolute;
    right: -3px;
    top: 50%;
    transform: translateY(-50%);
    width: 8px;
    height: 8px;
    background: #fff;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(91,94,244,0.8);
}

.conf-value {
    font-family: var(--mono);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--accent-light);
    white-space: nowrap;
    min-width: 52px;
    text-align: right;
}

/* Top-5 list */
.section-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.55rem;
}

.pred-row {
    display: flex;
    align-items: center;
    padding: 0.7rem 1rem;
    border-radius: var(--r-sm);
    margin-bottom: 0.3rem;
    gap: 0.8rem;
    background: var(--surface);
    border: 1px solid var(--border);
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
    cursor: default;
}

.pred-row:hover {
    border-color: var(--border-light);
    background: var(--surface-alt);
    transform: translateX(4px);
}

.pred-row.top {
    border-color: rgba(91,94,244,0.55);
    background: rgba(91,94,244,0.09);
    box-shadow: 0 0 24px rgba(91,94,244,0.1), inset 0 0 0 1px rgba(91,94,244,0.08);
}

.pred-rank {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    min-width: 22px;
}

.pred-row.top .pred-rank { color: var(--accent-light); }

.pred-chord {
    flex: 1;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    font-family: var(--sans);
}

.pred-bar-wrap {
    width: 72px;
    height: 3px;
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    overflow: hidden;
}

.pred-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent-light));
    opacity: 0.45;
}

.pred-row.top .pred-bar-fill { opacity: 1; box-shadow: 0 0 6px rgba(91,94,244,0.6); }

.pred-pct {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted);
    min-width: 44px;
    text-align: right;
}

.pred-row.top .pred-pct { color: var(--accent-light); }

/* Feature map section */
.feat-header {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.1rem 0 0.55rem;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-sec) !important;
    font-family: var(--sans) !important;
    padding: 0.75rem 1rem !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--surface-hi) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    border-left-width: 3px !important;
    font-family: var(--sans) !important;
    font-size: 0.84rem !important;
    background: var(--surface) !important;
}

/* Spinner */
[data-testid="stSpinner"] p {
    color: var(--text-muted) !important;
    font-family: var(--sans) !important;
    font-size: 0.84rem !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p {
    font-size: 0.71rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-family: var(--mono) !important;
}

/* Footer */
.app-footer {
    border-top: 1px solid var(--border);
    padding: 1.25rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}

.footer-left {
    font-size: 0.76rem;
    color: var(--text-muted);
    line-height: 1.6;
}

.footer-left strong { color: var(--text-sec); font-weight: 600; }

.footer-tags {
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
}

.ftag {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.22rem 0.5rem;
    border-radius: 5px;
    letter-spacing: 0.03em;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(91,94,244,0.45); }

/* Responsive */
@media (max-width: 920px) {
    .page-header { padding: 2.5rem 1.5rem 2rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        position: static;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border);
        padding: 1.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
        padding: 1.5rem !important;
    }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .chord-name { font-size: 2.8rem !important; }
    .pred-bar-wrap { display: none; }
    .app-footer { padding: 1.25rem 1.5rem; flex-direction: column; align-items: flex-start; }
}

@media (max-width: 500px) {
    .page-header { padding: 2rem 1rem 1.75rem; }
    .chord-display { padding: 1.5rem 1.25rem 1.25rem; }
    .chord-name { font-size: 2.4rem !important; }
    .page-title { font-size: 2rem !important; }
    .app-footer { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# -- Page header ---------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Chord<span class="grad">Net</span></h1>
    <p class="page-subtitle">
        Identifikasi chord piano secara otomatis menggunakan convolutional neural network
        dan tiga representasi fitur audio tingkat lanjut.
    </p>
</div>
""", unsafe_allow_html=True)

# -- Two-column layout ---------------------------------------------------------
col_left, col_right = st.columns([1, 1.9], gap="small")

# -- LEFT: input controls ------------------------------------------------------
with col_left:
    st.markdown('<div class="field-label">Sumber Audio</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Sumber audio",
        ["upload", "record"],
        format_func=lambda x: "Unggah File" if x == "upload" else "Rekam Langsung",
        label_visibility="collapsed",
        key="input_mode",
    )

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    if input_mode == "upload":
        st.markdown('<div class="field-label">File Audio</div>', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader(
            "Unggah audio",
            type=["wav", "mp3"],
            label_visibility="collapsed",
            help="Format WAV atau MP3 yang mengandung satu chord piano, idealnya 2\u20135 detik.",
        )
        if uploaded_audio is not None:
            st.audio(
                uploaded_audio,
                format="audio/wav" if uploaded_audio.name.lower().endswith(".wav") else "audio/mp3",
            )
        recorded_audio = None
    else:
        st.markdown('<div class="field-label">Rekaman Mikrofon</div>', unsafe_allow_html=True)
        if HAS_AUDIO_INPUT:
            recorded_audio = st.audio_input(
                "Rekam chord",
                label_visibility="collapsed",
            )
            if recorded_audio is not None:
                st.audio(recorded_audio, format="audio/wav")
            st.caption("Rekam hingga 10 detik. Sistem menggunakan 4 detik pertama untuk analisis.")
        else:
            st.warning(
                "Fitur rekam memerlukan Streamlit \u2265 1.31. "
                "Perbarui dengan `pip install -U streamlit`."
            )
            recorded_audio = None
        uploaded_audio = None

    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="field-label">Metode Analisis</div>', unsafe_allow_html=True)

    feature_key = st.radio(
        "Metode analisis",
        ["mel", "mfcc", "chroma"],
        format_func=lambda x: FEATURE_CONFIG[x]["display_name"],
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="desc-box">{FEATURE_CONFIG[feature_key]["description"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    run_button = st.button(
        "Deteksi Chord",
        type="primary",
        use_container_width=True,
    )

# -- RIGHT: results or empty state ---------------------------------------------
with col_right:
    if run_button:
        # Resolve active input source
        if input_mode == "upload":
            active_audio = uploaded_audio
            file_suffix  = (
                ".mp3"
                if (uploaded_audio is not None and uploaded_audio.name.lower().endswith(".mp3"))
                else ".wav"
            )
        else:
            active_audio = recorded_audio
            file_suffix  = ".wav"

        if active_audio is None:
            st.error("Silakan unggah file audio atau rekam chord terlebih dahulu.")
            st.session_state.result = None
        elif librosa is None:
            st.error("librosa belum terpasang. Pemrosesan audio tidak tersedia.")
            st.session_state.result = None
        else:
            with st.spinner("Memuat model\u2026"):
                try:
                    model, device = load_model(feature_key)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.session_state.result = None
                    st.stop()

            with st.spinner("Menganalisis audio\u2026"):
                try:
                    file_bytes = active_audio.read()
                    raw_feat   = extract_feature_from_audio(
                        file_bytes, feature_key, suffix=file_suffix,
                        denoise=(input_mode == "record"),
                    )
                    feature    = raw_feat[..., np.newaxis]
                    probs      = predict(feature, model, device)
                    top_idx    = np.argsort(probs)[::-1][:5]
                    st.session_state.result = {
                        "top_indices":   top_idx,
                        "probabilities": probs,
                        "feature":       feature,
                        "feature_key":   feature_key,
                    }
                except Exception as e:
                    st.error(f"Kesalahan pemrosesan: {e}")
                    st.session_state.result = None

    res = st.session_state.result

    if res is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">Belum ada hasil deteksi</div>
            <div class="empty-sub">Ikuti tiga langkah berikut untuk memulai analisis chord.</div>
            <div class="step-list">
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-text">Unggah file WAV/MP3 atau rekam chord langsung dari mikrofon</div>
                </div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-text">Pilih metode analisis fitur audio yang diinginkan</div>
                </div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-text">Tekan tombol <strong style="color:var(--text-sec);">Deteksi Chord</strong></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        top_indices   = res["top_indices"]
        probabilities = res["probabilities"]
        feature       = res["feature"]
        fkey          = res["feature_key"]

        top_label = format_label(LABELS[int(top_indices[0])])
        top_prob  = float(probabilities[int(top_indices[0])])
        bar_pct   = int(top_prob * 100)

        st.markdown(f"""
        <div class="chord-display">
            <div class="chord-meta">Chord Terdeteksi</div>
            <div class="chord-name">{top_label}</div>
            <div class="conf-row">
                <span class="conf-label">Keyakinan</span>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{bar_pct}%;"></div>
                </div>
                <span class="conf-value">{top_prob * 100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">5 Prediksi Teratas</div>', unsafe_allow_html=True)

        for rank, idx in enumerate(top_indices):
            chord  = format_label(LABELS[int(idx)])
            prob   = float(probabilities[int(idx)])
            pct    = int(prob * 100)
            is_top = "top" if rank == 0 else ""
            st.markdown(f"""
            <div class="pred-row {is_top}">
                <span class="pred-rank">#{rank + 1}</span>
                <span class="pred-chord">{chord}</span>
                <div class="pred-bar-wrap">
                    <div class="pred-bar-fill" style="width:{pct}%;"></div>
                </div>
                <span class="pred-pct">{prob * 100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        sq = feature.squeeze()
        st.markdown('<div class="feat-header">Peta Fitur Audio</div>', unsafe_allow_html=True)
        with st.expander(
            f"{FEATURE_CONFIG[fkey]['display_name']}  \u00b7  "
            f"{sq.shape[0]} \u00d7 {sq.shape[1]}  \u00b7  klik untuk tampilkan"
        ):
            render_feature_map(feature, fkey)
            st.caption(
                f"{FEATURE_CONFIG[fkey]['display_name']}  \u00b7  "
                f"{sq.shape[0]} freq bins \u00d7 {sq.shape[1]} time frames"
            )

# -- Footer --------------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div class="footer-left">
        <strong>ChordNet</strong> \u00b7 Sistem klasifikasi chord piano berbasis deep learning
    </div>
    <div class="footer-tags">
        <span class="ftag">PyTorch</span>
        <span class="ftag">librosa</span>
        <span class="ftag">CNN \u00b7 4 conv blocks</span>
        <span class="ftag">Mel \u00b7 MFCC \u00b7 Chroma</span>
        <span class="ftag">48 chord classes</span>
    </div>
</div>
""", unsafe_allow_html=True)
