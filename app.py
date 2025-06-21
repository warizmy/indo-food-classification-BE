import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel('gemini-pro')
    print(">>> Gemini AI berhasil dikonfigurasi.")
except Exception as e:
    llm_model = None
    print(f"Gagal konfigurasi Gemini AI. Fitur resep tidak akan berfungsi. Error: {e}")
    
try:
    model_path = os.path.join('models', 'MobileNetV2.h5')
    image_classifier_model = tf.keras.models.load_model(model_path)
    print(f">>> Model berhasil dimuat dari: {model_path}")
except Exception as e:
    image_classifier_model = None
    print(f"!!! KESALAHAN FATAL: Gagal memuat model dari '{model_path}'. Pastikan file ada. Error: {e}")

# Daftar kelas, pastikan urutannya sama persis dengan hasil training di Colab
CLASS_NAMES = [
    'Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng', 
    'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon', 'Rendang', 'Sate', 'Soto'
]

def process_image(image_path, target_size=(224, 224)):
    """Membuka, me-resize, dan memproses gambar untuk model."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        return None

def generate_recipe_from_llm(food_name):
    """Membuat prompt dan meminta resep dari Gemini AI."""
    if not llm_model:
        return "Maaf, fitur pembuatan resep sedang tidak tersedia."
    
    prompt = f"""
    Anda adalah seorang koki ahli masakan Indonesia.
    Berikan resep yang lezat dan mudah diikuti untuk masakan '{food_name}'.
    Gunakan format Markdown yang jelas, mencakup 'Deskripsi Singkat', 'Bahan-bahan', dan 'Langkah-langkah Pembuatan'.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error saat memanggil Gemini: {e}")
        return "Maaf, terjadi kesalahan saat mencoba membuat resep dari AI."

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint utama untuk prediksi gambar."""
    if image_classifier_model is None:
        return jsonify({'error': 'Model klasifikasi tidak tersedia di server.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'Request tidak menyertakan file gambar'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih untuk diunggah'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_image = process_image(filepath)
        if processed_image is None:
            return jsonify({'error': 'Format gambar tidak didukung atau file korup'}), 400

        prediction = image_classifier_model.predict(processed_image)
        food_name_raw = CLASS_NAMES[np.argmax(prediction)]
        food_name_display = food_name_raw.replace('_', ' ').title()
        recipe_text = generate_recipe_from_llm(food_name_display)

        # Buat URL lengkap yang bisa diakses dari mana saja oleh frontend
        base_url = request.host_url.replace('0.0.0.0', '127.0.0.1')
        image_url = f"{base_url}uploads/{filename}"

        return jsonify({
            'food_name': food_name_display,
            'recipe': recipe_text,
            'image_url': image_url
        })

    return jsonify({'error': 'Terjadi kesalahan tidak terduga'}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Endpoint untuk menyajikan/menampilkan gambar yang sudah di-upload."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # host='0.0.0.0' membuat server bisa diakses dari perangkat lain di jaringan yang sama
    app.run(debug=True, host='0.0.0.0', port=5000)
