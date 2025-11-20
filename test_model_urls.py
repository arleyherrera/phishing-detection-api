"""
Script de prueba para validar el modelo con URLs de ejemplo.

Prueba el mejor modelo entrenado con URLs legítimas y de phishing conocidas.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 80)
print("                 PRUEBA DEL MODELO DE DETECCIÓN DE PHISHING")
print("=" * 80)

# Cargar el mejor modelo
print("\n[1/4] Cargando modelo entrenado...")
best_model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')
print("[OK] Modelo Random Forest cargado")

# Cargar datos de validación (que tienen las features completas)
print("\n[2/4] Cargando datos de validación...")
X_val = joblib.load('data/processed/X_val.pkl')
y_val = joblib.load('data/processed/y_val.pkl')
print(f"[OK] Datos cargados: {len(X_val)} URLs")

# Tomar una muestra balanceada para la prueba
print("\n[3/4] Seleccionando muestra de prueba...")
np.random.seed(42)

# Obtener índices de legítimas y phishing
legitimate_indices = y_val[y_val == 0].index.tolist()
phishing_indices = y_val[y_val == 1].index.tolist()

# Tomar 10 de cada tipo
n_samples = 10
sample_legitimate = np.random.choice(legitimate_indices, n_samples, replace=False)
sample_phishing = np.random.choice(phishing_indices, n_samples, replace=False)

# Combinar
test_indices = np.concatenate([sample_legitimate, sample_phishing])
X_test = X_val.loc[test_indices]
y_test = y_val.loc[test_indices]

print(f"[OK] Muestra seleccionada:")
print(f"  - URLs legítimas: {n_samples}")
print(f"  - URLs phishing: {n_samples}")
print(f"  - Total: {len(test_indices)}")

# Realizar predicciones
print("\n[4/4] Realizando predicciones...")
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

print("[OK] Predicciones completadas")

# Resultados detallados
print("\n" + "=" * 80)
print("                           RESULTADOS DETALLADOS")
print("=" * 80)

# Mostrar cada predicción
df_results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicción': predictions,
    'Prob_Legítima': probabilities[:, 0],
    'Prob_Phishing': probabilities[:, 1],
})

df_results['Actual_Label'] = df_results['Actual'].map({0: 'Legítima', 1: 'Phishing'})
df_results['Pred_Label'] = df_results['Predicción'].map({0: 'Legítima', 1: 'Phishing'})
df_results['Correcto'] = df_results['Actual'] == df_results['Predicción']

print("\nPrimeras 10 URLs (Legítimas):")
print("-" * 80)
for i, row in df_results.head(10).iterrows():
    status = "[OK]" if row['Correcto'] else "[X]"
    print(f"{status} Real: {row['Actual_Label']:10} | Predicción: {row['Pred_Label']:10} | "
          f"Confianza: {row['Prob_Phishing']*100:.1f}% phishing")

print("\nSiguientes 10 URLs (Phishing):")
print("-" * 80)
for i, row in df_results.tail(10).iterrows():
    status = "[OK]" if row['Correcto'] else "[X]"
    print(f"{status} Real: {row['Actual_Label']:10} | Predicción: {row['Pred_Label']:10} | "
          f"Confianza: {row['Prob_Phishing']*100:.1f}% phishing")

# Métricas generales
print("\n" + "=" * 80)
print("                           MÉTRICAS DE RENDIMIENTO")
print("=" * 80)

accuracy = accuracy_score(y_test, predictions)
print(f"\n[OK] Accuracy: {accuracy*100:.2f}%")

# Matriz de confusión
cm = confusion_matrix(y_test, predictions)
print("\nMatriz de Confusión:")
print("-" * 40)
print(f"                 Predicho")
print(f"              Legítima  Phishing")
print(f"Real Legítima    {cm[0][0]:3d}      {cm[0][1]:3d}")
print(f"     Phishing    {cm[1][0]:3d}      {cm[1][1]:3d}")

# Reporte de clasificación
print("\nReporte de Clasificación:")
print("-" * 80)
print(classification_report(y_test, predictions,
                          target_names=['Legítima', 'Phishing'],
                          digits=4))

# Análisis de errores
errors = df_results[~df_results['Correcto']]
print("\n" + "=" * 80)
print("                           ANÁLISIS DE ERRORES")
print("=" * 80)

if len(errors) == 0:
    print("\n[!] PERFECTO! No se encontraron errores.")
    print("   Todas las URLs fueron clasificadas correctamente.")
else:
    print(f"\n[!] Se encontraron {len(errors)} errores:")
    print("-" * 80)
    for i, row in errors.iterrows():
        print(f"\n  URL índice: {i}")
        print(f"    Real: {row['Actual_Label']}")
        print(f"    Predicción: {row['Pred_Label']}")
        print(f"    Confianza phishing: {row['Prob_Phishing']*100:.2f}%")

    # Estadísticas de errores
    false_positives = len(errors[errors['Actual'] == 0])
    false_negatives = len(errors[errors['Actual'] == 1])

    print(f"\nDesglose de errores:")
    print(f"  - Falsos Positivos (legítimas → phishing): {false_positives}")
    print(f"  - Falsos Negativos (phishing → legítimas): {false_negatives}")

# Resumen final
print("\n" + "=" * 80)
print("                           RESUMEN FINAL")
print("=" * 80)

correct = df_results['Correcto'].sum()
total = len(df_results)

print(f"\nURLs analizadas: {total}")
print(f"Correctas: {correct} ({correct/total*100:.1f}%)")
print(f"Incorrectas: {total-correct} ({(total-correct)/total*100:.1f}%)")

if accuracy == 1.0:
    print("\n[***] EXCELENTE! El modelo tiene un rendimiento perfecto en esta muestra.")
elif accuracy >= 0.95:
    print("\n[OK] Buen rendimiento del modelo.")
elif accuracy >= 0.85:
    print("\n[!] Rendimiento aceptable, pero con margen de mejora.")
else:
    print("\n[X] Rendimiento bajo, requiere revision.")

print("\n" + "=" * 80)
print("                         PRUEBA COMPLETADA")
print("=" * 80)
