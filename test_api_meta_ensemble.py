"""
Test de la API con Meta-Ensemble usando URLs reales.
"""

import requests
import json

print("=" * 80)
print("  TEST: API con Meta-Ensemble - URLs Reales")
print("=" * 80)

API_URL = "http://localhost:5000/predict"

# URLs de prueba
test_urls = [
    # Legítimas con www
    ("https://www.google.com", "safe"),
    ("https://www.facebook.com", "safe"),
    ("https://www.bcp.com.pe", "safe"),
    ("https://www.sunat.gob.pe", "safe"),

    # Phishing
    ("http://xp0.936.mytemp.website", "phishing"),
    ("https://vodka1r.weebly.com/", "phishing"),
]

print(f"\n[1/2] Probando {len(test_urls)} URLs...")
print(f"API URL: {API_URL}\n")

results = []
for i, (url, expected) in enumerate(test_urls, 1):
    print(f"[{i}/{len(test_urls)}] {url[:60]}...")

    try:
        # Hacer request a la API
        response = requests.post(
            API_URL,
            json={"url": url},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            confidence = data['confidence']

            # Verificar si es correcto
            is_correct = (prediction == expected)
            status = "[OK]" if is_correct else "[X]"

            print(f"    {status} Predicción: {prediction:10s} | Confianza: {confidence*100:5.1f}% | Esperado: {expected}")

            results.append({
                'url': url,
                'expected': expected,
                'prediction': prediction,
                'confidence': confidence,
                'correct': is_correct
            })
        else:
            print(f"    [ERROR] Status code: {response.status_code}")
            print(f"    {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] {str(e)[:60]}")

# Resumen
print("\n" + "=" * 80)
print("  RESULTADOS")
print("=" * 80)

if results:
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nTotal URLs probadas: {total}")
    print(f"Correctas: {correct}/{total} ({accuracy:.1f}%)")

    if correct == total:
        print("\n[***] PERFECTO! Todas las predicciones correctas")
    else:
        print(f"\nErrores: {total - correct}")
        for r in results:
            if not r['correct']:
                print(f"  - {r['url']}: esperado {r['expected']}, obtenido {r['prediction']}")

print("\n" + "=" * 80)
