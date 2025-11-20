"""
Debug: Comparar features extraídas de URLs reales vs datos de entrenamiento.

Investigar por qué el modelo clasifica erróneamente sitios legítimos como phishing.
"""

import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('src')
from feature_extraction import URLFeatureExtractor

print("=" * 80)
print("  DEBUG: Análisis de Features - URLs Reales vs Entrenamiento")
print("=" * 80)

# Cargar datos de entrenamiento balanceados
print("\n[1/4] Cargando datos de entrenamiento...")
df_balanced = pd.read_csv('data/processed/balanced_phishtank_tranco.csv')

# Separar legítimas y phishing
df_train_legit = df_balanced[df_balanced['label'] == 0].copy()
df_train_phish = df_balanced[df_balanced['label'] == 1].copy()

print(f"[OK] Datos cargados:")
print(f"    - Legítimas entrenamiento: {len(df_train_legit):,}")
print(f"    - Phishing entrenamiento:  {len(df_train_phish):,}")

# URLs de prueba que fueron INCORRECTAMENTE clasificadas
print("\n[2/4] URLs reales MAL clasificadas (Legitimas -> Phishing):")
misclassified_urls = [
    'https://www.google.com',      # Predicho: phishing 59.2%
    'https://www.facebook.com',    # Predicho: phishing 74.4%
    'https://www.scotiabank.com.pe', # Predicho: phishing 96.4%
]

for url in misclassified_urls:
    print(f"    - {url}")

# URLs que fueron CORRECTAMENTE clasificadas
correctly_classified_legit = [
    'https://www.bcp.com.pe',      # Predicho: safe 97%
    'https://www.pucp.edu.pe',     # Predicho: safe 100%
]

print("\n[3/4] URLs reales BIEN clasificadas (Legitimas -> Safe):")
for url in correctly_classified_legit:
    print(f"    - {url}")

# Extraer features de URLs reales
print("\n[4/4] Extrayendo y comparando features...")

extractor = URLFeatureExtractor()
feature_names = joblib.load('models/fast_features_final.pkl')

# Función para extraer features consistentes
def extract_consistent_features(url, feature_names):
    """Extrae features y asegura que coincidan con feature_names"""
    features = extractor.extract_features(url)

    if not features:
        return None

    # Crear DataFrame con todas las features necesarias
    X = pd.DataFrame(0, index=[0], columns=feature_names)
    for feat in feature_names:
        if feat in features:
            X[feat] = features[feat]

    return X.iloc[0]

print("\n" + "=" * 80)
print("  COMPARACIÓN: URLs MAL CLASIFICADAS vs BIEN CLASIFICADAS")
print("=" * 80)

# Extraer features de URLs mal clasificadas
print("\n[A] Features de URLs MAL clasificadas (Google, Facebook, Scotiabank):")
misclassified_features = []
for url in misclassified_urls:
    print(f"\n  Procesando: {url}")
    feats = extract_consistent_features(url, feature_names)
    if feats is not None:
        misclassified_features.append({
            'url': url,
            'features': feats
        })
        # Mostrar features clave
        print(f"    url_length:        {feats.get('url_length', 0):.0f}")
        print(f"    domain_length:     {feats.get('domain_length', 0):.0f}")
        print(f"    num_dots:          {feats.get('num_dots', 0):.0f}")
        print(f"    num_hyphens:       {feats.get('num_hyphens', 0):.0f}")
        print(f"    num_digits:        {feats.get('num_digits', 0):.0f}")
        print(f"    num_subdomains:    {feats.get('num_subdomains', 0):.0f}")
        print(f"    has_https:         {feats.get('has_https', 0):.0f}")

# Extraer features de URLs bien clasificadas
print("\n[B] Features de URLs BIEN clasificadas (BCP, PUCP):")
correctly_features = []
for url in correctly_classified_legit:
    print(f"\n  Procesando: {url}")
    feats = extract_consistent_features(url, feature_names)
    if feats is not None:
        correctly_features.append({
            'url': url,
            'features': feats
        })
        # Mostrar features clave
        print(f"    url_length:        {feats.get('url_length', 0):.0f}")
        print(f"    domain_length:     {feats.get('domain_length', 0):.0f}")
        print(f"    num_dots:          {feats.get('num_dots', 0):.0f}")
        print(f"    num_hyphens:       {feats.get('num_hyphens', 0):.0f}")
        print(f"    num_digits:        {feats.get('num_digits', 0):.0f}")
        print(f"    num_subdomains:    {feats.get('num_subdomains', 0):.0f}")
        print(f"    has_https:         {feats.get('has_https', 0):.0f}")

# Comparar con datos de entrenamiento (sample)
print("\n[C] Features de URLs legítimas de ENTRENAMIENTO (Tranco sample):")

# Tomar 5 muestras aleatorias del entrenamiento
sample_train_legit = df_train_legit.sample(n=5, random_state=42)

for idx, row in sample_train_legit.iterrows():
    url = row.get('url', 'Unknown')
    print(f"\n  URL entrenamiento: {url[:60]}...")
    print(f"    url_length:        {row.get('url_length', 0):.0f}")
    print(f"    domain_length:     {row.get('domain_length', 0):.0f}")
    print(f"    num_dots:          {row.get('num_dots', 0):.0f}")
    print(f"    num_hyphens:       {row.get('num_hyphens', 0):.0f}")
    print(f"    num_digits:        {row.get('num_digits', 0):.0f}")
    print(f"    num_subdomains:    {row.get('num_subdomains', 0):.0f}")
    print(f"    has_https:         {row.get('has_https', 0):.0f}")

# Análisis estadístico
print("\n" + "=" * 80)
print("  ANÁLISIS ESTADÍSTICO")
print("=" * 80)

# Features numéricas para análisis
numeric_features = ['url_length', 'domain_length', 'num_dots', 'num_hyphens',
                   'num_digits', 'num_subdomains', 'has_https']

# Stats para entrenamiento (legítimas)
print("\n[1] Estadísticas de URLs LEGÍTIMAS de ENTRENAMIENTO:")
for feat in numeric_features:
    if feat in df_train_legit.columns:
        mean_val = df_train_legit[feat].mean()
        std_val = df_train_legit[feat].std()
        min_val = df_train_legit[feat].min()
        max_val = df_train_legit[feat].max()
        print(f"  {feat:20s}: mean={mean_val:6.1f}, std={std_val:6.1f}, min={min_val:6.1f}, max={max_val:6.1f}")

# Stats para entrenamiento (phishing)
print("\n[2] Estadísticas de URLs PHISHING de ENTRENAMIENTO:")
for feat in numeric_features:
    if feat in df_train_phish.columns:
        mean_val = df_train_phish[feat].mean()
        std_val = df_train_phish[feat].std()
        min_val = df_train_phish[feat].min()
        max_val = df_train_phish[feat].max()
        print(f"  {feat:20s}: mean={mean_val:6.1f}, std={std_val:6.1f}, min={min_val:6.1f}, max={max_val:6.1f}")

# Comparar features mal clasificadas con stats de entrenamiento
print("\n[3] Comparación: URLs mal clasificadas vs estadísticas de entrenamiento")

if len(misclassified_features) > 0:
    print("\n  Google.com comparado con legítimas de entrenamiento:")
    google_feats = misclassified_features[0]['features']

    for feat in numeric_features:
        if feat in df_train_legit.columns and feat in google_feats:
            train_mean = df_train_legit[feat].mean()
            train_std = df_train_legit[feat].std()
            google_val = google_feats[feat]

            # Calcular z-score (cuántas desviaciones estándar de la media)
            if train_std > 0:
                z_score = (google_val - train_mean) / train_std
            else:
                z_score = 0

            # Marcar si esta fuera de rango normal (|z| > 2)
            flag = "[!]" if abs(z_score) > 2 else "[OK]"

            print(f"  {flag} {feat:20s}: Google={google_val:6.1f}, Train_mean={train_mean:6.1f}, z-score={z_score:+.2f}")

print("\n" + "=" * 80)
print("  CONCLUSIONES")
print("=" * 80)

print("\nPosibles causas de falsos positivos:")
print("  1. Features mal extraídas de URLs reales (valores extremos)")
print("  2. Diferencias entre Tranco training data y URLs reales")
print("  3. Missing features (tld library warning)")
print("  4. Overfitting a patrones específicos de Tranco")

print("\n" + "=" * 80)
