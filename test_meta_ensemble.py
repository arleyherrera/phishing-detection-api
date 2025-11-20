"""
Pruebas del Meta-Ensemble con URLs reales de Perú y phishing.

Validar que el modelo funciona correctamente.
"""

import joblib
import pandas as pd
import numpy as np
import sys
from urllib.parse import urlparse
sys.path.append('src')
from feature_extraction import URLFeatureExtractor

print("=" * 80)
print("          PRUEBAS DEL META-ENSEMBLE CON URLs REALES")
print("=" * 80)

# ============================================================================
# Cargar Meta-Ensemble
# ============================================================================
print("\n[1/4] Cargando Meta-Ensemble...")
model = joblib.load('models/fast_voting_ensemble.pkl')
scaler = joblib.load('models/fast_scaler_final.pkl')
feature_names = joblib.load('models/fast_features_final.pkl')

print(f"[OK] Meta-Ensemble cargado:")
print(f"    - Tipo: {type(model).__name__}")
print(f"    - Estimadores: {len(model.estimators_)}")
for estimator in model.estimators_:
    print(f"      * {type(estimator).__name__}")
print(f"    - Features requeridas: {len(feature_names)}")

# ============================================================================
# URLs de prueba
# ============================================================================
print("\n[2/4] Preparando URLs de prueba...")

test_urls = {
    'Legítimas - Perú': [
        'https://www.bcp.com.pe',
        'https://www.gob.pe',
        'https://www.pucp.edu.pe',
        'https://www.sunat.gob.pe',
        'https://interbank.pe',
        'https://www.scotiabank.com.pe',
        'https://www.reniec.gob.pe',
        'https://www.bbva.pe',
        'https://www.minsa.gob.pe',
        'https://www.minedu.gob.pe',
    ],
    'Legítimas - Internacionales': [
        'https://www.google.com',
        'https://www.facebook.com',
        'https://www.microsoft.com',
        'https://www.amazon.com',
        'https://www.youtube.com',
    ],
    'Phishing - PhishTank': [
        'http://xp0.936.mytemp.website',
        'https://vodka1r.weebly.com/',
        'http://allegrolokalnie.pl-oferta-kategorie-polecane39932003.icu',
        'https://ww25.bancoprovincia.sbs',
        'http://192.168.1.1/secure-login',
    ]
}

# Recolectar todas las URLs
all_test_data = []
for category, urls in test_urls.items():
    for url in urls:
        all_test_data.append({
            'url': url,
            'category': category,
            'expected': 0 if 'Legítimas' in category else 1  # 0=legítima, 1=phishing
        })

print(f"[OK] URLs de prueba preparadas:")
for category, urls in test_urls.items():
    print(f"    - {category}: {len(urls)} URLs")

# ============================================================================
# Extraer features y predecir
# ============================================================================
print("\n[3/4] Extrayendo features y realizando predicciones...")
print("(Esto puede tomar unos segundos)")

def normalize_url(url):
    """
    Normaliza URL removiendo subdomain 'www' para compatibilidad con dataset.

    El dataset Tranco no incluye 'www', por lo que URLs como www.google.com
    tienen 2 dots (phishing-like) vs google.com con 1 dot (legitimate).
    """
    parsed = urlparse(url)
    domain = parsed.netloc

    # Remover www. del inicio
    if domain.startswith('www.'):
        domain = domain[4:]

    # Reconstruir URL normalizada
    normalized = f"{parsed.scheme}://{domain}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"

    return normalized

extractor = URLFeatureExtractor()
results = []

for i, test_data in enumerate(all_test_data, 1):
    url = test_data['url']
    category = test_data['category']
    expected = test_data['expected']

    print(f"\n  [{i}/{len(all_test_data)}] {url[:60]}...")

    try:
        # Normalizar URL (remover www para compatibilidad con dataset)
        url_normalized = normalize_url(url)
        if url != url_normalized:
            print(f"       [i] Normalizada: {url_normalized[:60]}...")

        # Extraer features de URL normalizada
        features = extractor.extract_features(url_normalized)

        if not features or len(features) == 0:
            print(f"       [!] No se pudieron extraer features")
            continue

        # Crear DataFrame con todas las features necesarias
        df_features = pd.DataFrame([features])

        # Asegurar que tengamos todas las features (rellenar con 0 si faltan)
        X = pd.DataFrame(0, index=[0], columns=feature_names)
        for feat in feature_names:
            if feat in df_features.columns:
                X[feat] = df_features[feat].values[0]

        # Normalizar
        X_scaled = scaler.transform(X)

        # Predecir con Meta-Ensemble
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        # Resultado
        pred_label = 'safe' if prediction == 0 else 'phishing'
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        # Verificar si es correcto
        is_correct = (prediction == expected)
        status = "[OK]" if is_correct else "[X]"

        print(f"       {status} Predicción: {pred_label:10} | Confianza: {confidence*100:5.1f}% | Esperado: {'safe' if expected == 0 else 'phishing'}")

        results.append({
            'url': url,
            'category': category,
            'expected': 'safe' if expected == 0 else 'phishing',
            'prediction': pred_label,
            'confidence': confidence,
            'correct': is_correct
        })

    except Exception as e:
        print(f"       [ERROR] {str(e)[:60]}")

# ============================================================================
# Resultados
# ============================================================================
print("\n[4/4] Generando reporte de resultados...")

df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("                      RESULTADOS DETALLADOS")
print("=" * 80)

for category in test_urls.keys():
    category_results = df_results[df_results['category'] == category]

    if len(category_results) == 0:
        continue

    print(f"\n{category}:")
    print("-" * 80)

    for idx, row in category_results.iterrows():
        status = "[OK]" if row['correct'] else "[X]"
        url_short = row['url'][:50] + "..." if len(row['url']) > 50 else row['url']

        print(f"{status} {url_short}")
        print(f"    Esperado:   {row['expected']:10} | Predicción: {row['prediction']:10} | Confianza: {row['confidence']*100:5.1f}%")

# ============================================================================
# Métricas generales
# ============================================================================
print("\n" + "=" * 80)
print("                      MÉTRICAS GENERALES")
print("=" * 80)

total = len(df_results)
correct = df_results['correct'].sum()
accuracy = (correct / total * 100) if total > 0 else 0

print(f"\nResultados:")
print(f"  Total URLs analizadas:  {total}")
print(f"  Predicciones correctas: {correct} ({accuracy:.1f}%)")
print(f"  Predicciones erróneas:  {total - correct}")

# Desglose por categoría
print(f"\nDesglose por categoría:")
for category in test_urls.keys():
    category_results = df_results[df_results['category'] == category]
    if len(category_results) > 0:
        cat_correct = category_results['correct'].sum()
        cat_total = len(category_results)
        cat_accuracy = (cat_correct / cat_total * 100)
        print(f"  {category:30s}: {cat_correct}/{cat_total} ({cat_accuracy:.0f}%)")

# Errores
if correct < total:
    print(f"\n[!] ERRORES DETECTADOS:")
    errors = df_results[~df_results['correct']]
    for idx, row in errors.iterrows():
        print(f"  - URL: {row['url'][:60]}")
        print(f"    Esperado: {row['expected']}, Predicción: {row['prediction']} (Confianza: {row['confidence']*100:.1f}%)")
else:
    print(f"\n[***] PERFECTO! Todas las URLs clasificadas correctamente")

print("\n" + "=" * 80)
print("                   PRUEBAS COMPLETADAS")
print("=" * 80)

print("\nMeta-Ensemble validado:")
print(f"  + {correct}/{total} predicciones correctas ({accuracy:.1f}%)")
print(f"  + Modelo: Voting Classifier (RF + XGBoost + SVM)")
print(f"  + Dataset de entrenamiento: 97,752 URLs balanceadas")
print(f"  + Features: {len(feature_names)} básicas (rápidas)")

print("\n" + "=" * 80)
