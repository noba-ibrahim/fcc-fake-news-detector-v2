"""
FCC Fake News Detector - API Flask Backend
Détection de fake news par Machine Learning
Version: 2.0 avec Architecture API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger données NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)

# CORS - Autoriser toutes les origines (à restreindre en production)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

print("="*60)
print("🚀 DÉMARRAGE API FLASK - FCC FAKE NEWS DETECTOR")
print("="*60)

# Chemins modèles
MODEL_PATH = os.path.join('models', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Vérifier fichiers
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle introuvable: {MODEL_PATH}")
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"❌ Vectoriseur introuvable: {VECTORIZER_PATH}")

# Charger modèle
print(f"📦 Chargement modèle...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print("✅ Modèle chargé")

# Charger vectoriseur
print(f"📦 Chargement vectoriseur...")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
print("✅ Vectoriseur chargé")

print("="*60)
print("✅ API PRÊTE À RECEVOIR DES REQUÊTES")
print("="*60 + "\n")

def clean_text(text):
    """Nettoyer le texte (identique notebook)"""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

@app.route('/')
def home():
    """Page d'accueil API"""
    return jsonify({
        'message': 'FCC Fake News Detector API',
        'version': '2.0',
        'status': 'active',
        'description': 'API ML de détection de fake news',
        'endpoints': {
            '/': 'GET - Infos API',
            '/health': 'GET - État santé',
            '/info': 'GET - Info modèle',
            '/predict': 'POST - Prédiction'
        },
        'github': 'https://github.com/noba-ibrahim/fcc-fake-news-detector'
    })

@app.route('/health')
def health():
    """Vérifier état API"""
    try:
        test_vector = vectorizer.transform(["test"])
        test_pred = model.predict(test_vector)
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'vectorizer_loaded': True,
            'test_ok': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/info')
def info():
    """Infos modèle"""
    return jsonify({
        'model': {
            'type': 'Logistic Regression',
            'accuracy': 98.34,
            'precision': 98.34,
            'recall': 98.34,
            'f1_score': 98.34
        },
        'vectorizer': {
            'type': 'TF-IDF',
            'max_features': 5000,
            'ngram_range': (1, 2),
            'vocabulary_size': len(vectorizer.vocabulary_)
        },
        'classes': {
            0: 'FAKE',
            1: 'REAL'
        },
        'dataset': {
            'total': 32456,
            'fake': 23481,
            'real': 8975
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédire si texte est fake ou real"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée'}), 400
        
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Champ "text" requis'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Texte trop court (min 10 car.)'}), 400
        
        if len(text) > 100000:
            return jsonify({'error': 'Texte trop long (max 100K car.)'}), 400
        
        print(f"\n📝 Prédiction: {len(text)} caractères")
        
        # Nettoyer
        clean = clean_text(text)
        
        # Vectoriser
        X = vectorizer.transform([clean])
        
        # Prédire
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        label = 'FAKE' if prediction == 0 else 'REAL'
        confidence = float(probabilities[prediction])
        
        print(f"✅ {label} ({confidence*100:.1f}%)")
        
        return jsonify({
            'prediction': int(prediction),
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'metadata': {
                'text_length': len(text),
                'clean_length': len(clean)
            }
        }), 200
    
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return jsonify({
            'error': 'Erreur prédiction',
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Route non trouvée',
        'available': ['/', '/health', '/info', '/predict']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Erreur serveur',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
