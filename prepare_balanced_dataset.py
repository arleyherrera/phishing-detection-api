"""
Preparar dataset BALANCEADO combinando:
- PhishTank (48,876 URLs phishing)
- Tranco Top 1M (48,876 URLs legítimas)

Total: ~97,752 URLs balanceadas 50/50
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from feature_extraction import URLFeatureExtractor
from tqdm import tqdm

print("=" * 80)
print("  PREPARACIÓN DE DATASET BALANCEADO: PhishTank + Tranco")
print("=" * 80)

# ============================================================================
# PASO 1: Cargar PhishTank (phishing)
# ============================================================================
print("\n[1/5] Cargando URLs de phishing de PhishTank...")
df_phishing_features = pd.read_csv('data/external/features_extracted_20251119_173932.csv')

# Filtrar solo phishing
df_phishing = df_phishing_features[df_phishing_features['label'] == 1].copy()

print(f"[OK] PhishTank phishing cargado:")
print(f"    - URLs phishing: {len(df_phishing):,}")
print(f"    - Features: {df_phishing.shape[1] - 2}")

# ============================================================================
# PASO 2: Cargar Tranco (legítimas)
# ============================================================================
print("\n[2/5] Cargando URLs legítimas de Tranco Top 1M...")
df_tranco = pd.read_csv('data/external/top-1m.csv', header=None, names=['rank', 'domain'])

# Tomar el mismo número de URLs que phishing para balance perfecto
n_legitimate_needed = len(df_phishing)

print(f"[OK] Tranco cargado:")
print(f"    - Total URLs disponibles: {len(df_tranco):,}")
print(f"    - URLs a usar (para balance): {n_legitimate_needed:,}")

# Tomar las primeras N (las más populares)
df_tranco_sample = df_tranco.head(n_legitimate_needed).copy()

# Agregar https://
df_tranco_sample['url'] = 'https://' + df_tranco_sample['domain']

print(f"[OK] URLs legítimas seleccionadas: {len(df_tranco_sample):,}")
print(f"\nEjemplos de URLs legítimas:")
for i, url in enumerate(df_tranco_sample['url'].head(10), 1):
    print(f"    {i:2d}. {url}")

# ============================================================================
# PASO 3: Extraer features de URLs Tranco
# ============================================================================
print(f"\n[3/5] Extrayendo features de {n_legitimate_needed:,} URLs legítimas...")
print("ADVERTENCIA: Esto tomará aproximadamente 10-15 minutos")
print("Progreso:")

extractor = URLFeatureExtractor()
legitimate_features_list = []

# Usar tqdm para barra de progreso
for idx, row in tqdm(df_tranco_sample.iterrows(), total=len(df_tranco_sample), desc="Extrayendo"):
    url = row['url']

    try:
        features = extractor.extract_features(url)
        if features and len(features) > 0:
            features['url'] = url
            features['label'] = 0  # Legítima
            legitimate_features_list.append(features)
        else:
            # Si no se pueden extraer features, usar valores por defecto
            legitimate_features_list.append({
                'url': url,
                'label': 0,
                'url_length': len(url),
                'domain_length': len(row['domain']),
            })
    except Exception as e:
        # En caso de error, continuar
        legitimate_features_list.append({
            'url': url,
            'label': 0,
            'url_length': len(url),
            'domain_length': len(row['domain']),
        })

df_legitimate = pd.DataFrame(legitimate_features_list)

print(f"\n[OK] Features extraídas de URLs legítimas:")
print(f"    - URLs procesadas: {len(df_legitimate):,}")
print(f"    - Features por URL: {df_legitimate.shape[1] - 2}")

# ============================================================================
# PASO 4: Combinar datasets
# ============================================================================
print("\n[4/5] Combinando datasets...")

# Asegurar que ambos tengan las mismas columnas
all_columns = list(set(df_phishing.columns) | set(df_legitimate.columns))

# Agregar columnas faltantes con 0
for col in all_columns:
    if col not in df_phishing.columns:
        df_phishing[col] = 0
    if col not in df_legitimate.columns:
        df_legitimate[col] = 0

# Ordenar columnas para que coincidan
df_phishing = df_phishing[all_columns]
df_legitimate = df_legitimate[all_columns]

# Combinar
df_combined = pd.concat([df_phishing, df_legitimate], ignore_index=True)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"[OK] Dataset combinado:")
print(f"    - Total URLs: {len(df_combined):,}")
print(f"    - Phishing: {(df_combined['label'] == 1).sum():,} ({(df_combined['label'] == 1).sum()/len(df_combined)*100:.1f}%)")
print(f"    - Legítimas: {(df_combined['label'] == 0).sum():,} ({(df_combined['label'] == 0).sum()/len(df_combined)*100:.1f}%)")
print(f"    - Features totales: {df_combined.shape[1] - 2}")

# ============================================================================
# PASO 5: Guardar dataset balanceado
# ============================================================================
print("\n[5/5] Guardando dataset balanceado...")

output_file = 'data/processed/balanced_phishtank_tranco.csv'
df_combined.to_csv(output_file, index=False)

print(f"[OK] Dataset guardado en:")
print(f"    {output_file}")
print(f"    Tamaño: {df_combined.shape}")

print("\n" + "=" * 80)
print("                    DATASET BALANCEADO LISTO")
print("=" * 80)

print(f"\nResumen:")
print(f"  ✅ {len(df_phishing):,} URLs de phishing (PhishTank)")
print(f"  ✅ {len(df_legitimate):,} URLs legítimas (Tranco Top 1M)")
print(f"  ✅ Balance perfecto: 50/50")
print(f"  ✅ {df_combined.shape[1] - 2} features extraídas")

print(f"\nPróximo paso:")
print(f"  Ejecutar: python train_fast_model_final.py")

print("\n" + "=" * 80)
