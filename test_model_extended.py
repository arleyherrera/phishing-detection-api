"""
Prueba extendida del modelo con 200 URLs (100 legitimas + 100 phishing).
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=" * 80)
print("              PRUEBA EXTENDIDA - 200 URLs (100 + 100)")
print("=" * 80)

# Cargar modelo
print("\nCargando modelo y datos...")
best_model = joblib.load('models/random_forest.pkl')
X_val = joblib.load('data/processed/X_val.pkl')
y_val = joblib.load('data/processed/y_val.pkl')

# Muestra grande
np.random.seed(42)
legitimate_indices = y_val[y_val == 0].index.tolist()
phishing_indices = y_val[y_val == 1].index.tolist()

n_samples = 100
sample_legitimate = np.random.choice(legitimate_indices, n_samples, replace=False)
sample_phishing = np.random.choice(phishing_indices, n_samples, replace=False)

test_indices = np.concatenate([sample_legitimate, sample_phishing])
X_test = X_val.loc[test_indices]
y_test = y_val.loc[test_indices]

print(f"[OK] Muestra: {len(test_indices)} URLs ({n_samples} legitimas + {n_samples} phishing)")

# Predicciones
print("\nRealizando predicciones...")
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

# Metricas
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("\n" + "=" * 80)
print("                           RESULTADOS")
print("=" * 80)

print(f"\nMETRICAS GENERALES:")
print(f"  Accuracy:  {accuracy*100:.2f}% ({int(accuracy*200)}/200 correctas)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Distribucion de confianza
print(f"\nDISTRIBUCION DE CONFIANZA:")
print(f"  Confianza promedio en phishing: {probabilities[n_samples:].mean()*100:.1f}%")
print(f"  Confianza promedio en legitimas: {probabilities[:n_samples].mean()*100:.1f}%")
print(f"  Confianza minima (legitimas): {probabilities[:n_samples].min()*100:.1f}%")
print(f"  Confianza maxima (phishing): {probabilities[n_samples:].max()*100:.1f}%")

# Errores
errors = (predictions != y_test).sum()
false_positives = ((predictions == 1) & (y_test == 0)).sum()
false_negatives = ((predictions == 0) & (y_test == 1)).sum()

print(f"\nERRORES:")
print(f"  Total de errores: {errors}")
print(f"  Falsos Positivos (legitima -> phishing): {false_positives}")
print(f"  Falsos Negativos (phishing -> legitima): {false_negatives}")

if errors == 0:
    print("\n[***] PERFECTO! 100% de accuracy en 200 URLs")
    print("      El modelo funciona EXCELENTEMENTE")
elif accuracy >= 0.99:
    print(f"\n[OK] Casi perfecto: {accuracy*100:.2f}%")
elif accuracy >= 0.95:
    print(f"\n[OK] Muy buen rendimiento: {accuracy*100:.2f}%")
else:
    print(f"\n[!] Rendimiento: {accuracy*100:.2f}%")

print("\n" + "=" * 80)
print("                    PRUEBA EXTENDIDA COMPLETADA")
print("=" * 80)
