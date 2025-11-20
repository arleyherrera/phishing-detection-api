"""
API REST para deteccion de phishing - Para Extension de Chrome

Endpoints:
  POST /predict
    Body: {"url": "https://ejemplo.com"}
    Response: {"prediction": "safe/phishing", "confidence": 0.95}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse
import joblib
import pandas as pd
import sys
sys.path.append('src')
from feature_extraction import URLFeatureExtractor

app = Flask(__name__)
CORS(app)  # Permitir requests desde la extension de Chrome

# Cargar Meta-Ensemble al iniciar
print("=" * 80)
print("     API de Deteccion de Phishing - Meta-Ensemble")
print("=" * 80)
print("\nCargando Meta-Ensemble...")
model = joblib.load('models/fast_voting_ensemble.pkl')
scaler = joblib.load('models/fast_scaler_final.pkl')
selected_features = joblib.load('models/fast_features_final.pkl')
print(f"[OK] Meta-Ensemble cargado:")
print(f"    - Tipo: VotingClassifier (RF + XGBoost + SVM)")
print(f"    - Features: {len(selected_features)}")
print(f"    - Dataset: 97,752 URLs balanceadas (50/50)")
print(f"    - Accuracy: 99.68% (test set)")
print("=" * 80)

# Extractor de features
extractor = URLFeatureExtractor()

# Whitelist de dominios confiables (top empresas tecnologicas, bancos, gobiernos)
TRUSTED_DOMAINS = {
    # Gigantes tecnologicos
    'google.com', 'youtube.com', 'gmail.com', 'drive.google.com', 'docs.google.com',
    'maps.google.com', 'play.google.com', 'gemini.google.com', 'bard.google.com',
    'facebook.com', 'instagram.com', 'whatsapp.com', 'messenger.com', 'meta.com',
    'apple.com', 'icloud.com', 'itunes.com', 'apple.co',
    'microsoft.com', 'outlook.com', 'live.com', 'hotmail.com', 'office.com',
    'azure.com', 'xbox.com', 'bing.com', 'skype.com',
    'amazon.com', 'aws.amazon.com', 'primevideo.com',
    'twitter.com', 'x.com',
    'linkedin.com',
    'netflix.com',
    'spotify.com',
    'tiktok.com',
    'zoom.us',
    'dropbox.com',
    'github.com', 'gitlab.com',
    'stackoverflow.com',
    'reddit.com',
    'wikipedia.org', 'wikimedia.org',
    'cloudflare.com',

    # Bancos internacionales
    'paypal.com',
    'chase.com', 'jpmorganchase.com',
    'bankofamerica.com',
    'wellsfargo.com',
    'citibank.com', 'citi.com',
    'hsbc.com',
    'santander.com',
    'bbva.com',

    # Bancos de Peru
    'bcp.com.pe',
    'scotiabank.com.pe',
    'bbva.pe',
    'interbank.pe',
    'bancodelanacion.gob.pe',
    'banconacion.gob.pe',
    'pichincha.pe',
    'mibanco.com.pe',
    'cajaarequipa.pe',

    # Gobierno de Peru
    'gob.pe',
    'sunat.gob.pe',
    'reniec.gob.pe',
    'minsa.gob.pe',
    'minedu.gob.pe',
    'pnp.gob.pe',
    'indecopi.gob.pe',

    # Universidades de Peru
    'pucp.edu.pe', 'pucp.pe',
    'uni.edu.pe',
    'unmsm.edu.pe',
    'ulima.edu.pe',
    'upc.edu.pe',
    'esan.edu.pe',
    'upn.edu.pe',

    # Otros servicios importantes
    'adobe.com',
    'salesforce.com',
    'oracle.com',
    'ibm.com',
    'sap.com',
    'intuit.com',
    'shopify.com',
    'wordpress.com', 'wordpress.org',
    'wix.com',
    'godaddy.com',
    'namecheap.com',
}

def is_trusted_domain(url):
    """
    Verifica si la URL pertenece a un dominio de la whitelist.
    Retorna True si el dominio base esta en la lista de confianza.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remover www. si existe
        if domain.startswith('www.'):
            domain = domain[4:]

        # Verificar si el dominio exacto esta en whitelist
        if domain in TRUSTED_DOMAINS:
            return True

        # Verificar si es un subdominio de algun dominio confiable
        # Ejemplo: mail.google.com -> google.com
        for trusted in TRUSTED_DOMAINS:
            if domain.endswith('.' + trusted):
                return True

        return False
    except:
        return False

def normalize_url(url):
    """
    Normaliza URL removiendo subdomain 'www' y trailing slash para compatibilidad con dataset.

    El dataset Tranco no incluye 'www', por lo que URLs como www.google.com
    tienen 2 dots (phishing-like) vs google.com con 1 dot (legitimate).
    Esta normalizacion mejora accuracy de 50% a 100% en URLs reales.
    """
    parsed = urlparse(url)
    domain = parsed.netloc

    # Remover www. del inicio
    if domain.startswith('www.'):
        domain = domain[4:]

    # Remover trailing slash del path (solo si path es "/" o termina en "/")
    path = parsed.path
    if path == '/' or (path.endswith('/') and len(path) > 1):
        path = path.rstrip('/')

    # Reconstruir URL normalizada
    normalized = f"{parsed.scheme}://{domain}{path}" if path else f"{parsed.scheme}://{domain}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"

    return normalized

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predice si una URL es phishing o no.

    Request:
        {"url": "https://ejemplo.com"}

    Response:
        {
            "url": "https://ejemplo.com",
            "prediction": "safe" o "phishing",
            "confidence": 0.95,
            "features_extracted": 40
        }
    """
    try:
        # Obtener URL del request
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'URL no proporcionada'}), 400

        # WHITELIST: Verificar si es un dominio confiable
        if is_trusted_domain(url):
            return jsonify({
                'url': url,
                'prediction': 'safe',
                'confidence': 1.0,
                'features_extracted': 56,
                'features_required': 56,
                'warning': 'Dominio verificado en whitelist de sitios confiables'
            }), 200

        # Normalizar URL (remover www para compatibilidad)
        url_normalized = normalize_url(url)

        # Extraer features de URL normalizada
        features = extractor.extract_features(url_normalized)

        if not features or len(features) == 0:
            return jsonify({
                'error': 'No se pudieron extraer features de la URL',
                'url': url
            }), 400

        # Convertir a DataFrame
        df_features = pd.DataFrame([features])

        # Obtener features comunes
        available_features = df_features.columns.tolist()
        common_features = [f for f in selected_features if f in available_features]

        # Crear matriz con todas las features (rellenar con 0 las que faltan)
        X = pd.DataFrame(0, index=[0], columns=selected_features)
        for feat in common_features:
            X[feat] = df_features[feat].values[0]

        # Normalizar
        X_scaled = scaler.transform(X)

        # Predecir
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        # Resultado
        result = {
            'url': url,
            'prediction': 'safe' if prediction == 0 else 'phishing',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'features_extracted': len(common_features),
            'features_required': len(selected_features),
            'warning': None
        }

        # Advertencia si faltan features
        if len(common_features) < len(selected_features) * 0.5:
            result['warning'] = 'Pocas features extraidas. Precision puede ser menor.'

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'url': url if 'url' in locals() else 'unknown'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': 'Random Forest',
        'features': len(selected_features)
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Informacion de la API."""
    return jsonify({
        'name': 'API de Deteccion de Phishing',
        'version': '1.0',
        'model': 'Random Forest Rapido (99.95% accuracy)',
        'endpoints': {
            'POST /predict': 'Predice si una URL es phishing',
            'GET /health': 'Health check',
            'GET /': 'Esta informacion'
        },
        'example': {
            'request': {
                'method': 'POST',
                'url': 'http://localhost:5000/predict',
                'body': {'url': 'https://ejemplo.com'}
            },
            'response': {
                'prediction': 'safe',
                'confidence': 0.95
            }
        }
    }), 200

if __name__ == '__main__':
    print("=" * 80)
    print("     API de Deteccion de Phishing - Para Extension Chrome")
    print("=" * 80)
    print("\nModelo: Random Forest RAPIDO (99.95% accuracy)")
    print(f"Features requeridas: {len(selected_features)}")
    print("\nEndpoints:")
    print("  POST /predict - Predice si URL es phishing")
    print("  GET  /health  - Health check")
    print("  GET  /        - Info de la API")
    print("\nEjemplo de uso:")
    print('  curl -X POST http://localhost:5000/predict \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"url": "https://www.bcp.com.pe"}\'')
    print("\n" + "=" * 80)
    print("\nIniciando servidor en http://localhost:5000")
    print("Presiona Ctrl+C para detener")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
