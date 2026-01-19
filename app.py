import os
import logging
from typing import Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['ENV'] = 'production'
app.config['DEBUG'] = False

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.path.join('models', 'MobileNetV2.h5')
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Food classification labels
CLASS_NAMES = [
    'Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
    'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
    'Rendang', 'Sate', 'Soto'
]


def initialize_gemini_ai() -> Optional[genai.GenerativeModel]:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='models/gemini-2.5-flash'
        )
        logger.info("Gemini AI successfully configured")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini AI: {str(e)}")
        return None


def load_classification_model() -> Optional[tf.keras.Model]:
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            return None
            
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Classification model successfully loaded from: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from '{MODEL_PATH}': {str(e)}")
        return None


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Image preprocessed successfully: {image_path}")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image '{image_path}': {str(e)}")
        return None


def generate_recipe(food_name: str, llm_model: Optional[genai.GenerativeModel]) -> str:
    if not llm_model:
        logger.warning("Recipe generation unavailable: LLM model not initialized")
        return "Recipe generation service is currently unavailable."
    
    prompt = f"""
    Generate a cooking recipe for Indonesian food: "{food_name}".

    STRICT RULES:
    - Write ONLY the recipe content.
    - Do NOT include greetings, introductions, or conclusions.
    - Do NOT mention yourself, the reader, or any role.
    - Do NOT add storytelling or conversational text.
    - Start directly with the recipe title.
    - Keep descriptions concise.
    - Avoid unnecessary culinary jargon.
    - Do NOT mention the food name over/on top the description. 

    FORMAT REQUIREMENTS:
    - Use Markdown.
    - Use the following EXACT, LITERAL structure:
    
    ### Deskripsi
    (Brief description of the dish, 2 sentences max)

    ### Bahan-bahan
    (List ingredients clearly using bullet points)

    ### Cara Membuat
    (Numbered step-by-step instructions)

    LANGUAGE:
    - Output must be in Bahasa Indonesia.
    - Use clear, concise, and practical cooking instructions.
    """
    
    try:
        response = llm_model.generate_content(prompt)
        logger.info(f"Recipe generated successfully for: {food_name}")
        return response.text
    except Exception as e:
        logger.error(f"Error generating recipe via Gemini AI: {str(e)}")
        return "An error occurred while generating the recipe. Please try again later."


llm_model = initialize_gemini_ai()
image_classifier_model = load_classification_model()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API status."""
    status = {
        'status': 'healthy',
        'classifier_model': image_classifier_model is not None,
        'recipe_generation': llm_model is not None
    }
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: file (image file)
    """
    
    if image_classifier_model is None:
        logger.error("Prediction request failed: Model not available")
        return jsonify({
            'error': 'Classification model is not available on the server'
        }), 503


    if 'file' not in request.files:
        logger.warning("Prediction request missing file")
        return jsonify({
            'error': 'No file provided in the request'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Prediction request with empty filename")
        return jsonify({
            'error': 'No file selected for upload'
        }), 400
    

    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        return jsonify({
            'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File uploaded successfully: {filename}")

        # Preprocess image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({
                'error': 'Invalid image format or corrupted file'
            }), 400

        # Perform prediction
        prediction = image_classifier_model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        food_name_raw = CLASS_NAMES[predicted_class_idx]
        food_name_display = food_name_raw.replace('_', ' ').title()
        
        logger.info(
            f"Prediction completed: {food_name_display} "
            f"(confidence: {confidence:.2%})"
        )

        # Generate recipe
        recipe_text = generate_recipe(food_name_display, llm_model)


        base_url = request.host_url.rstrip('/')
        if '0.0.0.0' in base_url:
            base_url = base_url.replace('0.0.0.0', '127.0.0.1')
        image_url = f"{base_url}/uploads/{filename}"

        return jsonify({
            'success': True,
            'food_name': food_name_display,
            'confidence': round(confidence, 4),
            'recipe': recipe_text,
            'image_url': image_url
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during processing'
        }), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file '{filename}': {str(e)}")
        return jsonify({'error': 'File not found'}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size too large error."""
    return jsonify({
        'error': 'File size exceeds maximum limit (16MB)'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error'
    }), 500