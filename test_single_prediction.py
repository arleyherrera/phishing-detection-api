"""
Test: Predicción de una sola URL con debug completo.
"""

import joblib
import pandas as pd
import sys
sys.path.append('src')
from feature_extraction import URLFeatureExtractor

print("=" * 80)
print("  TEST: Predicción de Google.com con Meta-Ensemble")
print("=" * 80)

# Cargar modelo
print("\n[1/4] Cargando modelo...")
model = joblib.load('models/fast_voting_ensemble.pkl')
scaler = joblib.load('models/fast_scaler_final.pkl')
feature_names = joblib.load('models/fast_features_final.pkl')
print(f"[OK] Modelo cargado")
print(f"     Features esperadas: {len(feature_names)}")

# URL de prueba
url = 'https://www.google.com'
print(f"\n[2/4] URL de prueba: {url}")

# Extraer features
print("\n[3/4] Extrayendo features...")
extractor = URLFeatureExtractor()
features = extractor.extract_features(url)

print(f"[OK] Features extraídas: {len(features)}")
print("\n  Primeras 20 features con valores:")
for i, (feat_name, feat_value) in enumerate(list(features.items())[:20], 1):
    print(f"    {i:2d}. {feat_name:30s} = {feat_value}")

# Crear DataFrame
df_features = pd.DataFrame([features])

# Asegurar que tengamos todas las features en el orden correcto
X = pd.DataFrame(0, index=[0], columns=feature_names)
for feat in feature_names:
    if feat in df_features.columns:
        X[feat] = df_features[feat].values[0]

print(f"\n[OK] DataFrame creado: {X.shape}")
print("\n  Primeras 20 features en DataFrame:")
for i, feat_name in enumerate(feature_names[:20], 1):
    print(f"    {i:2d}. {feat_name:30s} = {X[feat_name].values[0]}")

# Normalizar
print("\n[4/4] Normalizando y prediciendo...")
X_scaled = scaler.transform(X)

print(f"[OK] Features normalizadas")
print(f"\n  Primeras 10 features normalizadas:")
for i in range(min(10, len(X_scaled[0]))):
    print(f"    {i+1:2d}. {feature_names[i]:30s} = {X_scaled[0][i]:+.3f}")

# Predecir
prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

# Resultado
pred_label = 'safe' if prediction == 0 else 'phishing'
confidence = probabilities[1] if prediction == 1 else probabilities[0]

print("\n" + "=" * 80)
print("  RESULTADO")
print("=" * 80)
print(f"\nURL: {url}")
print(f"Predicción: {pred_label}")
print(f"Confianza: {confidence*100:.2f}%")
print(f"\nProbabilidades:")
print(f"  - Safe:     {probabilities[0]*100:.2f}%")
print(f"  - Phishing: {probabilities[1]*100:.2f}%")

# Predicciones individuales de cada estimador
print("\n" + "=" * 80)
print("  PREDICCIONES INDIVIDUALES")
print("=" * 80)

for estimator in model.estimators_:
    estimator_name = type(estimator).__name__
    pred = estimator.predict(X_scaled)[0]
    proba = estimator.predict_proba(X_scaled)[0]
    label = 'safe' if pred == 0 else 'phishing'
    conf = proba[1] if pred == 1 else proba[0]

    print(f"\n{estimator_name}:")
    print(f"  Predicción: {label}")
    print(f"  Confianza:  {conf*100:.2f}%")
    print(f"  Probas: safe={proba[0]*100:.2f}%, phishing={proba[1]*100:.2f}%")

print("\n" + "=" * 80)
