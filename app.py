from flask import Flask, render_template, request
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator, exceptions
from langdetect import detect, LangDetectException
from functools import lru_cache

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load Model & Vectorizer Safely
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CV_ACCURACY_PATH = "cv_results.txt"

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    logging.error("Model or vectorizer file is missing! Train it first.")
    raise FileNotFoundError("Model or vectorizer file is missing! Train it first.")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Model & Vectorizer Loaded Successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model or vectorizer.")

# Load Cross-Validation Results
cv_results = "Cross-validation results unavailable."
if os.path.exists(CV_ACCURACY_PATH):
    with open(CV_ACCURACY_PATH, "r") as f:
        cv_results = f.read()

@lru_cache(maxsize=500)
def translate_to_english(text):
    """Translate non-English text to English."""
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text[:4500])
        return text
    except (LangDetectException, exceptions.RequestError, exceptions.ElementNotFoundInGetRequest):
        return text  # Return original text if translation fails

def preprocess_text(text):
    """Clean and preprocess text, including language detection and translation."""
    if not text.strip():
        return ""
    text = translate_to_english(text)
    return text.lower()

def predict_news(news_article):
    """Predict whether the news is real or fake."""
    if not news_article.strip():
        return "⚠ Please enter a news article!"

    try:
        # Preprocess text
        processed_text = preprocess_text(news_article)
        article_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(article_tfidf)
        return "✅ This is Real News 📰" if prediction[0] == 0 else "❌ This is Fake News 🚨"
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Error processing the news article! Please try again."

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        news_article = request.form["news"].strip()
        result = predict_news(news_article) if news_article else "⚠ Please enter a news article!"
    return render_template("index.html", result=result, cv_results=cv_results)

if __name__ == "__main__":
    PORT = 5007
    while True:
        try:
            logging.info(f"🚀 Running Flask on http://127.0.0.1:{PORT}/")
            app.run(debug=True, host='127.0.0.1', port=PORT)
            break
        except OSError:
            logging.warning(f"⚠ Port {PORT} is in use. Trying next port...")
            PORT += 1
