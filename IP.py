# app.py
# Streamlit Image Processing Playground
# Methods: linear negative, contrast stretching, piecewise linear,
# log transform, gamma transform, histogram equalization, adaptive HE, CLAHE
# Works with grayscale and color images (per-channel or luminance-based where appropriate).

import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Optional dependency: scikit-image for true Adaptive HE (AHE)
try:
    from skimage import exposure as sk_exposure
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# ---------- Utilities ----------

def to_numpy_uint8(img_pil: Image.Image) -> np.ndarray:
    """PIL -> numpy uint8 RGB or grayscale."""
    if img_pil.mode in ["I;16", "I", "F"]:
        # Convert high-bit depth to 8-bit for display/processing
        img_pil = Image.fromarray(
            (np.array(img_pil, dtype=np.float32) / np.array(img_pil).max() * 255.0).astype(np.uint8)
        )
    if img_pil.mode == "RGBA":
        img_pil = img_pil.convert("RGB")
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    else:
        # Fallback: convert to RGB
        return np.array(img_pil.convert("RGB"))

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure grayscale uint8."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def is_grayscale(img: np.ndarray) -> bool:
    return img.ndim == 2

def clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

def normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / 255.0

def to_uint8_0_255(x: np.ndarray) -> np.ndarray:
    return clip_uint8(np.round(x * 255.0))

def plot_histogram(img: np.ndarray, title: str):
    """Plot histogram for grayscale or color (per-channel)."""
    fig, ax = plt.subplots()
    if img.ndim == 2:
        ax.hist(img.flatten(), bins=256, range=(0, 255))
        ax.set_title(f"{title} Histogram (Grayscale)")
        ax.set_xlim(0, 255)
    else:
        # RGB channels
        labels = ["R", "G", "B"]
        for i in range(3):
            ax.hist(img[:, :, i].flatten(), bins=256, range=(0, 255), alpha=0.5, label=labels[i])
        ax.legend()
        ax.set_title(f"{title} Histogram (RGB)")
        ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)

def show_side_by_side(before: np.ndarray, after: np.ndarray, caption_before="Before", caption_after="After"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.image(before, channels="RGB" if before.ndim==3 else "GRAY", caption=caption_before, use_container_width=True)
        plot_histogram(before, caption_before)
    with c2:
        st.image(after, channels="RGB" if after.ndim==3 else "GRAY", caption=caption_after, use_container_width=True)
        plot_histogram(after, caption_after)

def contrast_stretch(img: np.ndarray, in_lo: int, in_hi: int, out_lo: int, out_hi: int) -> np.ndarray:
    """Linear contrast stretching per channel. All values 0..255."""
    if in_hi <= in_lo:
        return img.copy()
    img = img.astype(np.float32)
    scale = (out_hi - out_lo) / (in_hi - in_lo)
    if img.ndim == 2:
        res = (img - in_lo) * scale + out_lo
    else:
        res = (img - in_lo) * scale + out_lo
    return clip_uint8(res)

def piecewise_linear_transform(img: np.ndarray, r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """
    Piecewise linear mapping defined by points (0,0) -> (r1,s1) -> (r2,s2) -> (255,255).
    Applies per channel independently.
    """
    r1 = np.clip(r1, 0, 255); r2 = np.clip(r2, 0, 255)
    s1 = np.clip(s1, 0, 255); s2 = np.clip(s2, 0, 255)
    if r2 <= r1:
        # invalid breakpoints; return unchanged
        return img.copy()

    def map_channel(ch):
        ch = ch.astype(np.float32)
        out = np.zeros_like(ch, dtype=np.float32)

        # Segment 1: [0, r1]
        idx1 = ch <= r1
        out[idx1] = (s1 / max(r1, 1e-6)) * ch[idx1]

        # Segment 2: (r1, r2]
        idx2 = (ch > r1) & (ch <= r2)
        out[idx2] = ((s2 - s1) / (r2 - r1)) * (ch[idx2] - r1) + s1

        # Segment 3: (r2, 255]
        idx3 = ch > r2
        out[idx3] = ((255 - s2) / (255 - r2)) * (ch[idx3] - r2) + s2

        return out

    if img.ndim == 2:
        return clip_uint8(map_channel(img))
    else:
        channels = [clip_uint8(map_channel(img[:, :, i])) for i in range(3)]
        return np.stack(channels, axis=2)

def log_transform(img: np.ndarray, c: float) -> np.ndarray:
    """s = c * log(1 + r), r in [0,1]; auto normalize to 0..255."""
    r = normalize_0_1(img)
    s = c * np.log1p(r)
    s = s / s.max() if s.max() > 0 else s
    return to_uint8_0_255(s)

def gamma_transform(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """s = c * r^gamma; normalized to 0..255."""
    r = normalize_0_1(img)
    s = c * np.power(r, gamma)
    s = s / s.max() if s.max() > 0 else s
    return to_uint8_0_255(s)

def hist_equalization(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization: grayscale directly; color on luminance (YCrCb)."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    # color: apply on Y channel to preserve colors
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def adaptive_hist_equalization(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    True AHE (no clipping). Requires scikit-image.
    If scikit-image isn't available, approximate with CLAHE at a very high clip limit.
    """
    if SKIMAGE_AVAILABLE:
        # skimage expects float in [0,1]; kernel_size is tile grid size per axis
        if img.ndim == 2:
            arr = normalize_0_1(img)
            out = sk_exposure.equalize_adapthist(arr, kernel_size=kernel_size, clip_limit=1.0, nbins=256)
            return to_uint8_0_255(out)
        else:
            # Do AHE on luminance
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            L = lab[:, :, 0] / 255.0
            L_eq = sk_exposure.equalize_adapthist(L, kernel_size=kernel_size, clip_limit=1.0, nbins=256)
            lab[:, :, 0] = L_eq * 255.0
            lab = lab.astype(np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Approximate using CLAHE with a large clip limit
        return clahe(img, clip_limit=40.0, tile_grid_size=kernel_size)

def clahe(img: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    clahe_op = cv2.createCLAHE(clipLimit=max(clip_limit, 0.0001), tileGridSize=(tile_grid_size, tile_grid_size))
    if img.ndim == 2:
        return clahe_op.apply(img)
    # color: on luminance
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe_op.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def linear_negative(img: np.ndarray) -> np.ndarray:
    return 255 - img

# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="Image Processing Playground",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Processing Playground")
st.caption("Choose a method, tweak parameters, and compare results with histograms.")

with st.sidebar:
    st.header("Upload & Input Options")
    uploaded = st.file_uploader("Upload an image (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    img_mode = st.radio("Interpret image as", ["Auto-detect", "Grayscale", "Color"], index=0)
    st.divider()
    method = st.selectbox(
        "Processing Method",
        [
            "Linear Negative",
            "Contrast Stretching",
            "Piecewise Linear Transformation",
            "Log Transformation",
            "Gamma Transformation",
            "Histogram Equalization",
            "Adaptive Histogram Equalization (AHE)",
            "CLAHE",
        ],
        index=0
    )

# Load image or fallback sample
if uploaded is not None:
    pil_img = Image.open(uploaded)
else:
    # Create a nice demo image: color gradient with a circle
    w, h = 512, 320
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)
    demo = np.zeros((h, w, 3), dtype=np.float32)
    demo[..., 0] = xv
    demo[..., 1] = yv
    demo[..., 2] = 0.5 * (np.sin(6 * np.pi * xv) * 0.5 + 0.5) + 0.25
    demo = np.clip(demo, 0, 1)
    pil_img = Image.fromarray((demo * 255).astype(np.uint8))

src = to_numpy_uint8(pil_img)

# Handle user-forced mode
if img_mode == "Grayscale":
    src_disp = ensure_gray(src)
elif img_mode == "Color":
    if src.ndim == 2:
        src_disp = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    else:
        src_disp = src
else:
    # Auto
    src_disp = src if src.ndim == 3 else src

# ---------- Parameter Panels & Processing ----------

params_col, _ = st.columns([1.2, 1], gap="large")
with params_col:
    st.subheader("Parameters")

# Defaults for processed image
processed = None

if method == "Linear Negative":
    st.markdown("Simple intensity inversion: `s = 255 - r`")
    processed = linear_negative(src_disp)

elif method == "Contrast Stretching":
    st.markdown("Map input range `[in_low, in_high]` to `[out_low, out_high]` linearly.")
    in_low, in_high = st.slider("Input range", 0, 255, (20, 230))
    out_low, out_high = st.slider("Output range", 0, 255, (0, 255))
    processed = contrast_stretch(src_disp, in_low, in_high, out_low, out_high)

elif method == "Piecewise Linear Transformation":
    st.markdown("Define two breakpoints (r1→s1, r2→s2) between (0→0) and (255→255).")
    r1 = st.slider("r1", 0, 255, 70)
    s1 = st.slider("s1", 0, 255, 50)
    r2 = st.slider("r2", 0, 255, 180)
    s2 = st.slider("s2", 0, 255, 210)
    processed = piecewise_linear_transform(src_disp, r1, s1, r2, s2)

elif method == "Log Transformation":
    st.markdown("Enhance dark regions: `s = c · log(1 + r)` with r∈[0,1].")
    c = st.slider("c (gain)", 0.1, 5.0, 1.0, 0.1)
    processed = log_transform(src_disp, c)

elif method == "Gamma Transformation":
    st.markdown("Power-law transform: `s = c · r^γ`, with automatic re-normalization.")
    gamma = st.slider("γ (gamma)", 0.10, 5.00, 0.50, 0.05)
    c = st.slider("c (gain)", 0.5, 2.0, 1.0, 0.05)
    processed = gamma_transform(src_disp, gamma, c)

elif method == "Histogram Equalization":
    st.markdown("Global histogram equalization (Y channel for color).")
    processed = hist_equalization(src_disp)

elif method == "Adaptive Histogram Equalization (AHE)":
    st.markdown(
        "Tile-based adaptive histogram equalization. "
        + ("(using scikit-image)" if SKIMAGE_AVAILABLE else "(approx. via CLAHE if scikit-image not available)")
    )
    tile = st.slider("Tile/Grid Size (kernel_size)", 4, 32, 8, 2)
    processed = adaptive_hist_equalization(src_disp, tile)

elif method == "CLAHE":
    st.markdown("Contrast Limited AHE (on luminance for color images).")
    clip_limit = st.slider("Clip Limit", 0.01, 10.0, 2.0, 0.01)
    tile = st.slider("Tile Grid Size", 4, 32, 8, 2)
    processed = clahe(src_disp, clip_limit, tile)

# ---------- Display ----------
st.subheader("Results")
show_side_by_side(
    src_disp if src_disp.ndim == 3 else src_disp,
    processed if processed is not None else src_disp,
    caption_before="Original",
    caption_after=f"{method}"
)

# ---------- Footer Tips ----------
with st.expander("Performance & Notes"):
    st.markdown(
        """
- All operations are fully vectorized NumPy/OpenCV to keep it fast even for large images.
- Color methods that change contrast are applied on luminance (Y in YCrCb or L in LAB) to avoid hue shifts.
- **AHE** uses `skimage.exposure.equalize_adapthist` when available; otherwise it falls back to a high-clip CLAHE approximation.
- Histograms are drawn from the displayed pixel values (uint8).
- If you upload a single-channel image but select **Color**, it will be promoted to RGB for visualization.
        """
    )
