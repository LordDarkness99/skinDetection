from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Mengizinkan Vercel (Frontend) untuk berkomunikasi dengan API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Nanti bisa diganti dengan URL Vercel Anda untuk keamanan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memuat model AI secara global saat server pertama kali menyala
print("Memuat model...")
model = tf.keras.models.load_model("model_kulit_terbaik.h5")
print("Model berhasil dimuat!")

# Daftar kelas sesuai urutan saat training
class_names = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis', 
    'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus', 
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 
    'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 
    'Vasculitis', 'Vitiligo', 'Warts'
]

def preprocess_image(image_bytes):
    # Membaca gambar, mengubah ke RGB, dan meresize ke 224x224
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    
    # Mengubah gambar menjadi array angka
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalisasi khusus MobileNetV2 (rentang -1 hingga 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Menambahkan dimensi batch (menjadi [1, 224, 224, 3])
    return tf.expand_dims(img_array, 0)

@app.get("/")
def home():
    return {"message": "API Deteksi Penyakit Kulit Aktif!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Membaca file gambar yang diunggah
        contents = await file.read()
        
        # Preprocessing
        processed_image = preprocess_image(contents)
        
        # Prediksi AI
        predictions = model.predict(processed_image)
        
        # Mengambil nilai tertinggi
        max_prob = np.max(predictions[0])
        class_idx = np.argmax(predictions[0])
        
        return {
            "status": "success",
            "prediction": class_names[class_idx],
            "confidence": float(max_prob) * 100 # Dalam persen
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}