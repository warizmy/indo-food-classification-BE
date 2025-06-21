# Indonesian Food Classification API

A Flask-based backend that classifies Indonesian food from an image and dynamically generates its recipe using the Google Gemini AI. This API is designed to be the intelligent core for any modern web or mobile application.

---

## Core Features

* **Image Classification**: Leverages a MobileNetV2-based model.
* **Dynamic Recipe Generation**: Integrates with the Google Gemini Pro API to provide relevant, high-quality recipes in real-time.
* **RESTful by Design**: Features a clean and simple `/predict` endpoint for seamless frontend integration.

---

## Tech Stack

-   **Language**: Python 3.12
-   **Framework**: Flask
-   **Machine Learning**: TensorFlow / Keras
-   **Generative AI**: Google Gemini Pro API
-   **Core Libraries**: Flask-CORS, Pillow, python-dotenv

---

## 🚀 Getting Started

To get the API running on your local machine, follow these steps.

### 1. Clone the Repository
```bash
git clone https://github.com/warizmy/indo-food-classification-BE.git
cd indo-food-classification-BE
```

### 2. Set Up a Virtual Environment
```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
All required packages are listed in ```requirements.txt```
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
The API requires a Google API key to function.
1. Create a new file named .env in the project's root directory.
2. Add your Google API key to the file:
```bash
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 5. Place the Model File
1. Create a ```models/``` directory in the project's root.
2. Place your trained model file inside this directory

## ▶️ Usage
Once the setup is complete, run the Flask server from your terminal:
```bash
python app.py
```
The API will be available at ```http://127.0.0.1:5000```

## API Endpoint
```POST /predict```
The primary endpoint for image classification.
- Method: ```POST```
- Body: ```multipart/form-data``` with a single key-value pair:
   - ```file```: The image file to be classified.
- Success Response (```200 OK```):
```bash
{
  "food_name": "Rendang",
  "image_url": "[http://127.0.0.1:5000/uploads/rendang.jpg](http://127.0.0.1:5000/uploads/rendang.jpg)",
  "recipe": "### Short Description\nRendang is a rich and tender coconut beef stew...\n..."
}
```
- Error Response:
```bash
{
  "error": "Request does not contain an image file"
}
```
