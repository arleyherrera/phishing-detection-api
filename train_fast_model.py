"""
Entrenar modelo RAPIDO para API en tiempo real.

Este modelo usa solo features BASICAS de URL que se pueden extraer
instantaneamente (sin descargar HTML).

Para usar en extension de Chrome + API.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("     ENTRENAMIENTO DE MODELO RAPIDO PARA API (Features Basicas)")
print("=" * 80)

# Cargar datos con features basicas extraidas
print("\n[1/5] Cargando datos de PhishTank con features basicas...")
df = pd.read_csv('data/external/features_extracted_20251119_173932.csv')

print(f"[OK] Datos cargados: {len(df)} URLs")
print(f"    - Features: {df.shape[1] - 2} (excluyendo url y label)")
print(f"    - Phishing: {(df['label'] == 1).sum()}")
print(f"    - Legitimas: {(df['label'] == 0).sum()}")

# Preparar features y labels
print("\n[2/5] Preparando datos...")
feature_cols = [col for col in df.columns if col not in ['url', 'label']]
X = df[feature_cols]
y = df['label']

print(f"[OK] Features para el modelo: {len(feature_cols)}")
print(f"    Primeras 10: {feature_cols[:10]}")

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[OK] Division de datos:")
print(f"    - Train: {len(X_train)} URLs")
print(f"    - Test: {len(X_test)} URLs")

# Normalizar
print("\n[3/5] Normalizando features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features normalizadas")

# Entrenar modelo
print("\n[4/5] Entrenando Random Forest...")
print("    (esto puede tomar 2-3 minutos)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("[OK] Modelo entrenado")

# Evaluar
print("\n[5/5] Evaluando modelo...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 80)
print("                    RESULTADOS DEL MODELO RAPIDO")
print("=" * 80)

print(f"\nMetricas en Test Set ({len(X_test)} URLs):")
print(f"  - Accuracy:  {accuracy*100:.2f}%")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall:    {recall:.4f}")
print(f"  - F1-Score:  {f1:.4f}")

# Matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusion:")
print(f"                 Predicho")
print(f"              Legitima  Phishing")
print(f"Real Legitima    {cm[0][0]:5d}    {cm[0][1]:5d}")
print(f"     Phishing    {cm[1][0]:5d}    {cm[1][1]:5d}")

# Guardar modelo
print("\n" + "=" * 80)
print("                    GUARDANDO MODELO")
print("=" * 80)

joblib.dump(model, 'models/fast_model.pkl')
joblib.dump(scaler, 'models/fast_scaler.pkl')
joblib.dump(feature_cols, 'models/fast_features.pkl')

print("\n[OK] Modelo guardado:")
print("    - models/fast_model.pkl")
print("    - models/fast_scaler.pkl")
print("    - models/fast_features.pkl")

print("\n" + "=" * 80)
print("                    MODELO LISTO PARA API")
print("=" * 80)

print("\nEste modelo:")
print(f"  + Usa {len(feature_cols)} features basicas de URL")
print("  + NO requiere descargar HTML")
print("  + Respuesta INSTANTANEA (<100ms)")
print(f"  + Precision: {accuracy*100:.1f}%")
print("\nListo para usar en:")
print("  - Extension de Chrome")
print("  - API REST")
print("  - Deteccion en tiempo real")

print("\n" + "=" * 80)
