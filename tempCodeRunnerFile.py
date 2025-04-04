import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from deep_translator import GoogleTranslator, exceptions
from langdetect import detect, LangDetectException
from functools import lru_cache
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv(r"C:\Users\91630\Desktop\FRI dataset.csv", low_memory=False)

# Rename columns to expected format
df.rename(columns={"Label": "label", "ArticleBody": "text"}, inplace=True)

# Validate columns
expected_columns = {"label", "text"}
actual_columns = set(df.columns)
if not expected_columns.issubset(actual_columns):
    raise KeyError(f"Dataset must contain 'label' and 'text' columns, but found {actual_columns}")

# Load stopwords
stop_words_en = set(stopwords.words('english'))
stop_words_hi = {"अभी", "अगर", "और", "भी", "लेकिन", "द्वारा", "है", "था", "यह", "उनका", "हम", "क्या", "के", "को", "किया", "साथ", "तक"}

@lru_cache(maxsize=500)
def translate_to_english(text):
    """Translate non-English text to English safely."""
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text[:4500])  # Avoid exceeding 5000-char limit
        return text
    except (LangDetectException, exceptions.RequestError, exceptions.ElementNotFoundInGetRequest):
        return text  # Return original text if translation fails

def preprocess_text(text):
    """Clean and preprocess text."""
    if pd.isna(text) or not text.strip():
        return ""
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'
    
    stop_words = stop_words_en if lang == 'en' else stop_words_hi if lang == 'hi' else set()
    text = translate_to_english(text) if lang != 'en' else text
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    return ' '.join(word for word in text.split() if word not in stop_words)

# Preprocess data
df['text'] = df['text'].astype(str).apply(preprocess_text)

# Extract Features and Labels
X = df['text']
y = df['label']  # Keeping original labels without swapping

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42, stratify=y)

# Use TfidfVectorizer for feature extraction
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model using 5-Fold Cross-Validation
model = RandomForestClassifier(n_estimators=15, max_depth=3, min_samples_split=30, random_state=42)

cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation Accuracy Scores: {cv_scores}')
print(f'Average Accuracy: {np.mean(cv_scores):.4f}')

# Train final model on training set
model.fit(X_train_tfidf, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", report)

# Save model and vectorizer
pickle.dump(model, open("best_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save cross-validation results
with open("cv_results.txt", "w") as f:
    f.write(f'Cross-validation Accuracy Scores: {cv_scores}\n')
    f.write(f'Average Accuracy: {np.mean(cv_scores):.4f}\n')
    f.write(f'Classification Report:\n{report}\n')

# Feature importance analysis
if hasattr(model, 'feature_importances_'):
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 Important Features:")
    with open("cv_results.txt", "a") as f:
        f.write("Top 10 Important Features:\n")
    
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
        with open("cv_results.txt", "a") as f:
            f.write(f"{feature}: {importance:.4f}\n")
