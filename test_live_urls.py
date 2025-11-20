"""
Prueba REAL con URLs en vivo - Extraccion completa de features.

Este script:
1. Toma URLs reales (phishing de PhishTank + legitimas de Peru)
2. Descarga el HTML de cada URL
3. Extrae las 40 features del modelo UCI
4. Hace predicciones con el modelo entrenado
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
import time
from feature_extraction import URLFeatureExtractor
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("        PRUEBA REAL CON URLs EN VIVO - EXTRACCION COMPLETA")
print("=" * 80)

# URLs de prueba
test_urls = {
    'Legitimas Peru': [
        'https://www.bcp.com.pe',
        'https://www.gob.pe',
        'https://www.pucp.edu.pe',
    ],
    'Phishing PhishTank': [
        'http://xp0.936.mytemp.website',
        'https://vodka1r.weebly.com/',
        'https://tuya365info.frappe.cloud/loggin',
    ]
}

all_test_urls = []
expected_labels = []
categories = []

for category, urls in test_urls.items():
    all_test_urls.extend(urls)
    if 'Legitima' in category:
        expected_labels.extend([0] * len(urls))  # 0 = Legitima
    else:
        expected_labels.extend([1] * len(urls))  # 1 = Phishing
    categories.extend([category] * len(urls))

print(f"\n[1/5] URLs seleccionadas para prueba: {len(all_test_urls)}")
print("\nLegitimas (Peru):")
for url in test_urls['Legitimas Peru']:
    print(f"  [OK] {url}")

print("\nPhishing (PhishTank):")
for url in test_urls['Phishing PhishTank']:
    print(f"  [!] {url}")

# Cargar modelo
print("\n[2/5] Cargando modelo entrenado...")
best_model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')
selected_features = joblib.load('data/processed/selected_features.pkl')
print(f"[OK] Modelo cargado: Random Forest")
print(f"[OK] Features requeridas: {len(selected_features)}")

# Extraer features (esto toma tiempo)
print("\n[3/5] Extrayendo features de URLs en vivo...")
print("NOTA: Esto puede tomar 1-2 minutos por URL (descarga HTML + analisis)")
print("-" * 80)

extractor = URLFeatureExtractor()
extracted_features_list = []

for i, url in enumerate(all_test_urls, 1):
    print(f"\n[{i}/{len(all_test_urls)}] Procesando: {url[:60]}...")
    start_time = time.time()

    try:
        # Extraer features (esto descarga la pagina)
        features = extractor.extract_features(url)
        elapsed = time.time() - start_time

        if features and len(features) > 0:
            extracted_features_list.append(features)
            print(f"      [OK] Extraidas {len(features)} features en {elapsed:.1f}s")
        else:
            print(f"      [!] No se pudieron extraer features (URL inaccesible?)")
            # Features por defecto (URL basica)
            extracted_features_list.append({
                'URLLength': len(url),
                'DomainLength': len(url.split('/')[2]) if len(url.split('/')) > 2 else 0,
            })

    except Exception as e:
        print(f"      [X] Error: {str(e)[:60]}")
        # Features por defecto en caso de error
        extracted_features_list.append({
            'URLLength': len(url),
            'DomainLength': len(url.split('/')[2]) if len(url.split('/')) > 2 else 0,
        })

print("\n[OK] Extraccion completada")

# Convertir a DataFrame
df_features = pd.DataFrame(extracted_features_list)
df_features['url'] = all_test_urls
df_features['expected_label'] = expected_labels
df_features['category'] = categories

print(f"\nFeatures extraidas: {df_features.shape}")
print(f"Primeras features: {list(df_features.columns[:5])}")

# Preparar para prediccion
print("\n[4/5] Preparando features para prediccion...")

# Verificar que features tenemos
available_features = [col for col in df_features.columns
                     if col not in ['url', 'expected_label', 'category']]
print(f"Features disponibles: {len(available_features)}")

# Features comunes con el modelo
common_features = [f for f in selected_features if f in available_features]
print(f"Features comunes con modelo: {len(common_features)} de {len(selected_features)}")

if len(common_features) < 10:
    print("\n[!] ADVERTENCIA: Pocas features comunes con el modelo UCI")
    print("    Las predicciones pueden no ser precisas.")
    print("    Esto es porque el modelo UCI requiere features HTML avanzadas")
    print("    que solo se pueden extraer descargando completamente la pagina.")
    print("\n    Para demostracion, usaremos las features disponibles.")

# Crear matriz de features
# Si tenemos pocas features comunes, rellenar con 0
X_test = pd.DataFrame(0, index=range(len(df_features)), columns=selected_features)

# Llenar con las features que si tenemos
for feat in common_features:
    X_test[feat] = df_features[feat].values

print(f"[OK] Matriz de features preparada: {X_test.shape}")

# Normalizar
try:
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("[OK] Features normalizadas")
except Exception as e:
    print(f"[!] Error al normalizar: {e}")
    X_test_scaled = X_test

# Predicciones
print("\n[5/5] Realizando predicciones...")
predictions = best_model.predict(X_test_scaled)
probabilities = best_model.predict_proba(X_test_scaled)

print("[OK] Predicciones completadas")

# Resultados
print("\n" + "=" * 80)
print("                         RESULTADOS DETALLADOS")
print("=" * 80)

df_results = pd.DataFrame({
    'URL': all_test_urls,
    'Categoria': categories,
    'Esperado': expected_labels,
    'Prediccion': predictions,
    'Prob_Phishing': probabilities[:, 1],
    'Prob_Legitima': probabilities[:, 0],
})

df_results['Esperado_Label'] = df_results['Esperado'].map({0: 'Legitima', 1: 'Phishing'})
df_results['Pred_Label'] = df_results['Prediccion'].map({0: 'Legitima', 1: 'Phishing'})
df_results['Correcto'] = df_results['Esperado'] == df_results['Prediccion']

print("\nURLs Legitimas de Peru:")
print("-" * 80)
legitimas = df_results[df_results['Categoria'] == 'Legitimas Peru']
for idx, row in legitimas.iterrows():
    status = "[OK]" if row['Correcto'] else "[X]"
    url_short = row['URL'][:50] + "..." if len(row['URL']) > 50 else row['URL']
    print(f"{status} {url_short}")
    print(f"    Esperado: {row['Esperado_Label']:10} | Prediccion: {row['Pred_Label']:10}")
    print(f"    Confianza Phishing: {row['Prob_Phishing']*100:.1f}% | Legitima: {row['Prob_Legitima']*100:.1f}%")

print("\nURLs Phishing de PhishTank:")
print("-" * 80)
phishing = df_results[df_results['Categoria'] == 'Phishing PhishTank']
for idx, row in phishing.iterrows():
    status = "[OK]" if row['Correcto'] else "[X]"
    url_short = row['URL'][:50] + "..." if len(row['URL']) > 50 else row['URL']
    print(f"{status} {url_short}")
    print(f"    Esperado: {row['Esperado_Label']:10} | Prediccion: {row['Pred_Label']:10}")
    print(f"    Confianza Phishing: {row['Prob_Phishing']*100:.1f}% | Legitima: {row['Prob_Legitima']*100:.1f}%")

# Metricas
print("\n" + "=" * 80)
print("                           METRICAS FINALES")
print("=" * 80)

correctas = df_results['Correcto'].sum()
total = len(df_results)
accuracy = correctas / total

print(f"\nResultados:")
print(f"  URLs analizadas: {total}")
print(f"  Correctas: {correctas} ({accuracy*100:.1f}%)")
print(f"  Incorrectas: {total - correctas}")

# Errores
if correctas == total:
    print("\n[***] PERFECTO! Todas las URLs clasificadas correctamente")
else:
    errores = df_results[~df_results['Correcto']]
    print(f"\n[!] Errores encontrados:")
    for idx, row in errores.iterrows():
        print(f"  - {row['URL'][:50]}")
        print(f"    Esperado: {row['Esperado_Label']}, Predicho: {row['Pred_Label']}")

print("\n" + "=" * 80)
print("                    PRUEBA CON URLs REALES COMPLETADA")
print("=" * 80)

# Nota final
print("\nNOTA IMPORTANTE:")
if len(common_features) < len(selected_features):
    print(f"Solo se pudieron extraer {len(common_features)} de {len(selected_features)} features.")
    print("Para precision maxima, se necesita descargar y analizar completamente")
    print("el HTML, CSS, JavaScript y certificados SSL de cada pagina.")
    print("Esto requiere 30-60 segundos por URL y conexion estable.")
