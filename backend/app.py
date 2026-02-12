"""
API Flask pour la Détection de Fake News
Modèle : Random Forest Optimized
Version : 2.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)  # Permettre les requêtes depuis d'autres domaines

print("=" * 60)
print("🚀 FCC FAKE NEWS DETECTOR API - DÉMARRAGE")
print("=" * 60)

# ============================================================
# CHARGEMENT DES MODÈLES ML
# ============================================================

print("\n📦 Chargement des modèles depuis models/...")

# Charger le modèle Random Forest
try:
    model_path = os.path.join('models', 'random_forest_optimized.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé : Random Forest Optimized")
except FileNotFoundError:
    print(f"❌ ERREUR : Fichier '{model_path}' non trouvé")
    print("   → Vérifiez que le fichier existe dans backend/models/")
    model = None
except Exception as e:
    print(f"❌ ERREUR lors du chargement du modèle : {e}")
    model = None

# Charger le vectorizer TF-IDF
try:
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✅ Vectorizer chargé : TF-IDF")
except FileNotFoundError:
    print(f"❌ ERREUR : Fichier '{vectorizer_path}' non trouvé")
    print("   → Vérifiez que le fichier existe dans backend/models/")
    vectorizer = None
except Exception as e:
    print(f"❌ ERREUR lors du chargement du vectorizer : {e}")
    vectorizer = None

# Vérification finale
if model is not None and vectorizer is not None:
    print("\n" + "=" * 60)
    print("✅ TOUS LES MODÈLES SONT CHARGÉS AVEC SUCCÈS")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("⚠️  ATTENTION : MODÈLES NON CHARGÉS")
    print("=" * 60)

# ============================================================
# FONCTION DE NETTOYAGE DU TEXTE
# ============================================================

def clean_text(text):
    """
    Nettoie et prétraite le texte avant l'analyse
    
    Étapes :
    1. Conversion en minuscules
    2. Suppression des URLs
    3. Suppression des emails
    4. Suppression des caractères spéciaux
    5. Suppression des stopwords
    6. Lemmatisation
    
    Args:
        text (str): Texte brut à nettoyer
    
    Returns:
        str: Texte nettoyé et prétraité
    """
    try:
        # 1. Minuscules
        text = text.lower()
        
        # 2. Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 3. Supprimer les emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Garder uniquement lettres et espaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 5. Supprimer espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Supprimer les stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 7. Lemmatisation
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        # Retourner le texte nettoyé
        cleaned = ' '.join(words)
        
        return cleaned
    
    except Exception as e:
        print(f"❌ Erreur dans clean_text : {e}")
        return text

# ============================================================
# ROUTES DE L'API
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """
    Page d'accueil de l'API
    
    Retourne les informations sur l'API et les endpoints disponibles
    """
    return jsonify({
        "message": "FCC Fake News Detector API",
        "version": "2.0",
        "model": "Random Forest Optimized",
        "status": "operational" if (model and vectorizer) else "models not loaded",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Fake news prediction (POST)"
        },
        "author": "FCC Development Team",
        "year": 2024
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """
    Vérification de l'état de santé de l'API
    
    Vérifie que les modèles sont chargés correctement
    """
    if model is not None and vectorizer is not None:
        return jsonify({
            "status": "healthy",
            "model": "loaded",
            "vectorizer": "loaded",
            "model_type": "Random Forest Optimized"
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "model": "not loaded" if model is None else "loaded",
            "vectorizer": "not loaded" if vectorizer is None else "loaded",
            "error": "Models failed to load"
        }), 500


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """
    Endpoint principal pour la prédiction de fake news
    
    Reçoit un texte en JSON et retourne la prédiction
    
    Body JSON:
        {
            "text": "Article text to analyze..."
        }
    
    Retourne JSON:
        {
            "prediction": 0 ou 1,
            "label": "REAL" ou "FAKE",
            "confidence": 0.0-1.0,
            "probabilities": {
                "real": 0.0-1.0,
                "fake": 0.0-1.0
            },
            "text_length": int,
            "cleaned_length": int
        }
    """
    
    # Vérifier que les modèles sont chargés
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Models not loaded",
            "message": "ML models failed to load. Please check server logs."
        }), 500
    
    try:
        # 1. Récupérer les données de la requête
        data = request.get_json()
        
        # Vérifier que le champ 'text' existe
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request",
                "message": "Please provide a 'text' field in JSON body"
            }), 400
        
        text = data['text']
        
        # Vérifier que le texte n'est pas vide
        if not text or len(text.strip()) < 10:
            return jsonify({
                "error": "Text too short",
                "message": "Please provide at least 10 characters"
            }), 400
        
        print(f"\n📝 Nouvelle requête reçue : {len(text)} caractères")
        
        # 2. Nettoyer le texte
        cleaned_text = clean_text(text)
        
        # Vérifier que le texte nettoyé n'est pas vide
        if not cleaned_text or len(cleaned_text) < 5:
            return jsonify({
                "error": "Text cleaning resulted in empty string",
                "message": "Text contains no meaningful content after preprocessing"
            }), 400
        
        print(f"🧹 Texte nettoyé : {len(cleaned_text)} caractères")
        
        # 3. Vectoriser le texte
        text_vectorized = vectorizer.transform([cleaned_text])
        print(f"🔢 Texte vectorisé : shape {text_vectorized.shape}")
        
        # 4. Faire la prédiction
        prediction = int(model.predict(text_vectorized)[0])
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # 5. Préparer la réponse
        result = {
            "prediction": prediction,
            "label": "FAKE" if prediction == 1 else "REAL",
            "confidence": float(max(probabilities)),
            "probabilities": {
                "real": float(probabilities[0]),
                "fake": float(probabilities[1])
            },
            "text_length": len(text),
            "cleaned_length": len(cleaned_text),
            "model": "Random Forest Optimized"
        }
        
        print(f"✅ Prédiction : {result['label']} (Confiance : {result['confidence']:.2%})")
        print("-" * 60)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An error occurred during prediction",
            "details": str(e)
        }), 500


# ============================================================
# ROUTE DE TEST (OPTIONNEL)
# ============================================================

@app.route('/test', methods=['GET'])
def test():
    """
    Route de test pour vérifier rapidement l'API
    """
    test_text = "Breaking news! Scientists discovered shocking truth that doctors don't want you to know!"
    
    try:
        cleaned = clean_text(test_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = int(model.predict(vectorized)[0])
        probs = model.predict_proba(vectorized)[0]
        
        return jsonify({
            "test": "success",
            "sample_text": test_text[:50] + "...",
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence": float(max(probs)),
            "message": "API is working correctly"
        }), 200
    
    except Exception as e:
        return jsonify({
            "test": "failed",
            "error": str(e)
        }), 500


# ============================================================
# DÉMARRAGE DU SERVEUR
# ============================================================

if __name__ == '__main__':
    # Récupérer le port depuis les variables d'environnement
    # (Render définit automatiquement PORT)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 Démarrage du serveur sur le port {port}...")
    print(f"📍 URL locale : http://0.0.0.0:{port}")
    print(f"🔗 Endpoints disponibles :")
    print(f"   - GET  /         → Informations API")
    print(f"   - GET  /health   → Vérification santé")
    print(f"   - POST /predict  → Prédiction fake news")
    print(f"   - GET  /test     → Test rapide")
    print("=" * 60 + "\n")
    
    # Lancer l'application
    app.run(
        host='0.0.0.0',        # Écouter sur toutes les interfaces
        port=port,
        debug=False,           # False en production
        use_reloader=False     # False en production
    )
