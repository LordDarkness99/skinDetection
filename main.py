import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import keras


# PATCH: Custom layer legacy dari MobileNetV2

@keras.utils.register_keras_serializable(package="Custom")
class TrueDivide(keras.layers.Layer):
    """Legacy: x / 127.5"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x / 127.5

    def __call__(self, *args, **kwargs):
        tensors = [a for a in args if hasattr(a, "shape")]
        if not tensors:
            tensors = list(args)
        return super().__call__(tensors[0], **kwargs)

    def get_config(self):
        return super().get_config()


@keras.utils.register_keras_serializable(package="Custom")
class Subtract(keras.layers.Layer):
    """Legacy: x - 1.0"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x - 1.0

    def __call__(self, *args, **kwargs):
        tensors = [a for a in args if hasattr(a, "shape")]
        if not tensors:
            tensors = list(args)
        return super().__call__(tensors[0], **kwargs)

    def get_config(self):
        return super().get_config()



# KONFIGURASI HALAMAN & APPLE-LIKE STYLING (CSS)

st.set_page_config(
    page_title="DermaAI — Deteksi Kulit",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Container Padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1024px;
    }

    /* Apple-style Hero Title */
    .hero-title {
        font-size: 2.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1d1d1f;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        color: #86868b;
        margin-bottom: 2rem;
    }

    /* Glass / Clean Card styling */
    .apple-card {
        background-color: #fbfbfd;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }

    .apple-metric-card {
        background-color: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.02);
    }

    /* Custom Badges */
    .badge-high {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Progress bar smooth rounding */
    stProgress > div > div > div > div {
        border-radius: 999px;
    }
    
    /* Expander Apple clean style */
    details {
        background: #fbfbfd;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px !important;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)



# KAMUS 22 KELAS PENYAKIT KULIT

CLASS_INFO = {
    "Acne": {"desc": "Peradangan kulit akibat sumbatan kelenjar minyak dan bakteri.", "urgensi": "Rendah-Sedang"},
    "Actinic_Keratosis": {"desc": "Lesi pra-kanker akibat paparan sinar UV kronis.", "urgensi": "Tinggi (Wajib Cek)"},
    "Benign_tumors": {"desc": "Tumor kulit jinak (non-kanker) yang umumnya tidak berbahaya.", "urgensi": "Rendah"},
    "Bullous": {"desc": "Kelainan kulit ditandai dengan terbentuknya lepuhan atau melepuh.", "urgensi": "Sedang-Tinggi"},
    "Candidiasis": {"desc": "Infeksi jamur Candida pada lipatan kulit atau area lembap.", "urgensi": "Sedang"},
    "DrugEruption": {"desc": "Reaksi alergi obat bermanifestasi pada kulit.", "urgensi": "Sedang-Tinggi"},
    "Eczema": {"desc": "Dermatitis atopik/alergi, kulit kering, merah, dan sangat gatal.", "urgensi": "Sedang"},
    "Infestations_Bites": {"desc": "Gigitan atau infestasi parasit (kutu, tungau, serangga).", "urgensi": "Sedang"},
    "Lichen": {"desc": "Kelainan inflamasi kronis pada kulit/selaput lendir (Lichen Planus).", "urgensi": "Sedang"},
    "Lupus": {"desc": "Manifestasi kulit dari penyakit autoimun sistemik.", "urgensi": "Tinggi"},
    "Moles": {"desc": "Nevus melanositik (tahi lalat normal/atipikal).", "urgensi": "Pantau Berkala"},
    "Psoriasis": {"desc": "Penyakit autoimun membuat sel kulit tumbuh cepat bersisik tebal.", "urgensi": "Sedang-Tinggi"},
    "Rosacea": {"desc": "Kemerahan kronis pada wajah disertai pembuluh darah kecil tampak.", "urgensi": "Sedang"},
    "Seborrh_Keratoses": {"desc": "Pertumbuhan jinak permukaan kulit menyerupai lilin/menempel.", "urgensi": "Rendah"},
    "SkinCancer": {"desc": "Keganasan kulit (melanoma/non-melanoma).", "urgensi": "Kritis (Segera ke Dokter)"},
    "Sun_Sunlight_Damage": {"desc": "Kerusakan struktural kulit akibat akumulasi sinar matahari.", "urgensi": "Sedang"},
    "Tinea": {"desc": "Infeksi jamur superficial (kurap / kadas).", "urgensi": "Sedang"},
    "Unknown_Normal": {"desc": "Kondisi kulit normal atau tidak terklasifikasi spesifik.", "urgensi": "Normal"},
    "Vascular_Tumors": {"desc": "Tumor pembuluh darah jinak/abnormal (hemangioma dll).", "urgensi": "Sedang"},
    "Vasculitis": {"desc": "Peradangan pada pembuluh darah kulit.", "urgensi": "Tinggi"},
    "Vitiligo": {"desc": "Kehilanganpigmen kulit sehingga tampak bercak putih.", "urgensi": "Sedang (Kosmetik)"},
    "Warts": {"desc": "Kutil akibat infeksi Human Papillomavirus (HPV) jinak.", "urgensi": "Rendah-Sedang"},
}



# PATH & LOAD MODEL
BASE        = Path(__file__).parent if "__file__" in locals() else Path(".")
MODEL_PATH  = BASE / "model_kulit_terbaik.h5"
LABELS_PATH = BASE / "labels.json"
IMG_SIZE    = (224, 224)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        custom_objects={"TrueDivide": TrueDivide, "Subtract": Subtract},
    )
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
    return model

@st.cache_data(show_spinner=False)
def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return list(CLASS_INFO.keys())



# SIDEBAR: DIREKTORI 22 KELAS
with st.sidebar:
    st.markdown("### 📚 Direktori 22 Kelas")
    st.caption("Referensi cepat jenis kondisi kulit yang didukung model.")
    search_q = st.text_input("Cari kelas...", placeholder="Misal: Acne, Psoriasis...")
    
    st.divider()
    container_kb = st.container(height=420)
    with container_kb:
        for idx, (cls_name, info) in enumerate(CLASS_INFO.items(), 1):
            if search_q.lower() in cls_name.lower() or search_q.lower() in info["desc"].lower():
                with st.expander(f"{idx:02d}. {cls_name.replace('_', ' ')}"):
                    st.write(f"**Info:** {info['desc']}")
                    st.caption(f"**Urgensi Klinis:** {info['urgensi']}")



# MAIN APP HEADER
st.markdown('<div class="hero-title">DermaAI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Analisis citra klinis berbasis deep learning dengan standar presisi tinggi.</div>', unsafe_allow_html=True)

# Load model & labels
model  = load_model()
LABELS = load_labels()

if model is None:
    st.error(f"⚠️ File model tidak ditemukan di `{MODEL_PATH.name}`. Pastikan file model berada di direktori yang sama.")
    st.stop()



# INPUT SECTION (TAB APPLE STYLE)
tab_cam, tab_up = st.tabs(["📷 Tangkap Kamera", "🖼️ Unggah Berkas"])
image = None

with tab_cam:
    cam = st.camera_input("Ambil foto area kulit")
    if cam is not None:
        image = Image.open(cam)

with tab_up:
    up = st.file_uploader("Seret atau pilih gambar (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        image = Image.open(up)



# PREDICTION & RESULTS
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

def predict(img: Image.Image):
    probs = model.predict(preprocess(img), verbose=0)[0]
    idx = np.argsort(probs)[::-1]
    # Safety guard if labels length matches
    safe_labels = LABELS if len(LABELS) == len(probs) else list(CLASS_INFO.keys())[:len(probs)]
    return [(safe_labels[i], float(probs[i])) for i in idx]

if image is not None:
    st.markdown("---")
    c1, c2 = st.columns([1.1, 1.3], gap="large")

    with c1:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.caption("INPUT GAMBAR")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Menganalisis pola dermatologis..."):
        results = predict(image)

    top_label, top_conf = results[0]
    clean_top_name = top_label.replace('_', ' ')
    info_dict = CLASS_INFO.get(top_label, {"desc": "Analisis model AI terdeteksi.", "urgensi": "Konsultasikan dokter"})

    with c2:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.caption("HASIL PREDIKSI UTAMA")
        
        # Badge keyakinan
        conf_pct = top_conf * 100
        badge_class = "badge-high" if conf_pct >= 75 else ("badge-moderate" if conf_pct >= 45 else "badge-low")
        st.markdown(f'<span class="{badge_class}">Tingkat Keyakinan: {conf_pct:.1f}%</span>', unsafe_allow_html=True)
        
        st.markdown(f"## {clean_top_name}")
        st.write(info_dict["desc"])
        
        st.info(f"🛡️ **Rekomendasi Urgensi:** {info_dict['urgensi']}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Top 5 Breakdown Card
    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
    st.subheader("Distribusi Top 5 Kandidat")
    for label, conf in results[:5]:
        clean_name = label.replace('_', ' ')
        pct = conf * 100
        cols_r = st.columns([2, 5, 1])
        with cols_r[0]:
            st.markdown(f"**{clean_name}**")
        with cols_r[1]:
            st.progress(float(conf))
        with cols_r[2]:
            st.caption(f"{pct:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer gaya Apple Health
    st.warning(
        "**Catatan Medis Penting:** Utilitas ini berbasis AI eksperimental dan bukan perangkat medis resmi. "
        "Segera temui Dokter Spesialis Kulit & Kelamin (Sp.KK / Sp.DV) untuk diagnosis akurat."
    )
