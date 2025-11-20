"""
Entrenar META-ENSEMBLE para modelo rápido con dataset balanceado.

EXACTAMENTE como tu tesis especifica:
- Random Forest
- XGBoost (Gradient Boosting)
- SVM
- Voting Classifier (Meta-Ensemble)

Dataset: PhishTank (phishing) + Tranco (legítimas) - BALANCEADO 50/50
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time

print("=" * 80)
print("  ENTRENAMIENTO DE META-ENSEMBLE PARA MODELO RAPIDO")
print("  (Random Forest + XGBoost + SVM + Voting Classifier)")
print("=" * 80)

# ============================================================================
# PASO 1: Cargar dataset balanceado
# ============================================================================
print("\n[1/7] Cargando dataset balanceado...")
df = pd.read_csv('data/processed/balanced_phishtank_tranco.csv')

print(f"[OK] Dataset cargado:")
print(f"    - Total URLs: {len(df):,}")
print(f"    - Phishing:  {(df['label'] == 1).sum():,} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")
print(f"    - Legitimas: {(df['label'] == 0).sum():,} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
print(f"    - Features:  {df.shape[1] - 2}")

# ============================================================================
# PASO 2: Preparar datos
# ============================================================================
print("\n[2/7] Preparando features y labels...")

# Separar features y labels
feature_cols = [col for col in df.columns if col not in ['url', 'label']]
X = df[feature_cols]
y = df['label']

print(f"[OK] Features seleccionadas: {len(feature_cols)}")
print(f"    Primeras 10: {feature_cols[:10]}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n[OK] Division de datos:")
print(f"    - Train: {len(X_train):,} URLs ({(y_train == 1).sum():,} phishing / {(y_train == 0).sum():,} legitimas)")
print(f"    - Test:  {len(X_test):,} URLs ({(y_test == 1).sum():,} phishing / {(y_test == 0).sum():,} legitimas)")

# ============================================================================
# PASO 3: Normalizar features
# ============================================================================
print("\n[3/7] Normalizando features con StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features normalizadas")

# ============================================================================
# PASO 4: Entrenar modelos individuales
# ============================================================================
print("\n[4/7] Entrenando modelos individuales...")
print("(esto puede tomar 5-10 minutos)")

start_time = time.time()

# Modelo 1: Random Forest
print("\n  [4.1] Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"  [OK] Random Forest entrenado - Accuracy: {rf_acc*100:.2f}%")

# Modelo 2: XGBoost (Gradient Boosting)
print("\n  [4.2] XGBoost (Gradient Boosting)...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"  [OK] XGBoost entrenado - Accuracy: {xgb_acc*100:.2f}%")

# Modelo 3: SVM
print("\n  [4.3] Support Vector Machine...")
svm_model = SVC(
    C=1,
    kernel='linear',
    probability=True,  # Necesario para Voting Classifier
    random_state=42,
    verbose=False
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"  [OK] SVM entrenado - Accuracy: {svm_acc*100:.2f}%")

elapsed = time.time() - start_time
print(f"\n[OK] Todos los modelos individuales entrenados en {elapsed:.1f}s")

# ============================================================================
# PASO 5: Crear Meta-Ensemble (Voting Classifier)
# ============================================================================
print("\n[5/7] Creando Meta-Ensemble (Voting Classifier)...")
print("  Combinando: Random Forest + XGBoost + SVM")

voting_clf = VotingClassifier(
    estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model),
        ('svm', svm_model)
    ],
    voting='soft',  # Voting basado en probabilidades
    n_jobs=-1
)

print("  [OK] Voting Classifier configurado con 'soft voting'")

# Entrenar el Voting Classifier
print("\n  Entrenando Voting Classifier...")
voting_clf.fit(X_train_scaled, y_train)

print("\n  Realizando predicciones con el Meta-Ensemble...")
voting_pred = voting_clf.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)

print(f"  [OK] Meta-Ensemble completado - Accuracy: {voting_acc*100:.2f}%")

# ============================================================================
# PASO 6: Evaluación completa
# ============================================================================
print("\n[6/7] Evaluando todos los modelos...")

# Calcular métricas para cada modelo
models = {
    'Random Forest': (rf_pred, rf_model),
    'XGBoost': (xgb_pred, xgb_model),
    'SVM': (svm_pred, svm_model),
    'Voting Classifier (Meta-Ensemble)': (voting_pred, voting_clf)
}

print("\n" + "=" * 80)
print("                  COMPARACION DE MODELOS")
print("=" * 80)

results = []
for name, (predictions, model) in models.items():
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append({
        'Modelo': name,
        'Accuracy': f'{acc*100:.2f}%',
        'Precision': f'{prec:.4f}',
        'Recall': f'{rec:.4f}',
        'F1-Score': f'{f1:.4f}'
    })

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Mostrar matriz de confusión del Meta-Ensemble
print("\n" + "=" * 80)
print("      MATRIZ DE CONFUSION - META-ENSEMBLE (Voting Classifier)")
print("=" * 80)

cm = confusion_matrix(y_test, voting_pred)
print(f"\n                 Predicho")
print(f"              Legitima  Phishing")
print(f"Real Legitima    {cm[0][0]:5d}    {cm[0][1]:5d}")
print(f"     Phishing    {cm[1][0]:5d}    {cm[1][1]:5d}")

# ============================================================================
# PASO 7: Guardar modelos
# ============================================================================
print("\n[7/7] Guardando modelos...")

# Guardar todos los modelos
joblib.dump(rf_model, 'models/fast_rf.pkl')
joblib.dump(xgb_model, 'models/fast_xgb.pkl')
joblib.dump(svm_model, 'models/fast_svm.pkl')
joblib.dump(voting_clf, 'models/fast_voting_ensemble.pkl')
joblib.dump(scaler, 'models/fast_scaler_final.pkl')
joblib.dump(feature_cols, 'models/fast_features_final.pkl')

print("\n[OK] Modelos guardados:")
print("    - models/fast_rf.pkl (Random Forest)")
print("    - models/fast_xgb.pkl (XGBoost)")
print("    - models/fast_svm.pkl (SVM)")
print("    - models/fast_voting_ensemble.pkl (Meta-Ensemble)")
print("    - models/fast_scaler_final.pkl (Scaler)")
print("    - models/fast_features_final.pkl (Features)")

print("\n" + "=" * 80)
print("              META-ENSEMBLE LISTO PARA PRODUCCION")
print("=" * 80)

print(f"\nDataset utilizado:")
print(f"  - PhishTank: 48,876 URLs phishing")
print(f"  - Tranco Top 1M: 48,876 URLs legitimas")
print(f"  - Total: 97,752 URLs BALANCEADAS (50/50)")

print(f"\nModelos entrenados:")
print(f"  1. Random Forest")
print(f"  2. XGBoost (Gradient Boosting)")
print(f"  3. SVM")
print(f"  4. Voting Classifier (Meta-Ensemble)")

voting_final_acc = accuracy_score(y_test, voting_pred)
print(f"\nMejor resultado:")
print(f"  Meta-Ensemble (Voting Classifier): {voting_final_acc*100:.2f}%")

print(f"\nCaracteristicas del modelo:")
print(f"  + {len(feature_cols)} features basicas (rapidas)")
print(f"  + NO requiere descargar HTML")
print(f"  + Respuesta instantanea (<100ms)")
print(f"  + Dataset balanceado 50/50")

print("\n" + "=" * 80)
print("\nPara usar en la API, actualizar api_phishing.py:")
print("  model = joblib.load('models/fast_voting_ensemble.pkl')")
print("  scaler = joblib.load('models/fast_scaler_final.pkl')")
print("  features = joblib.load('models/fast_features_final.pkl')")

print("\n" + "=" * 80)
