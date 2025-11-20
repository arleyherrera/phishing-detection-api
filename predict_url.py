"""
predict_url.py

Script de inferencia para detección de phishing en producción.

Uso:
    python predict_url.py <URL>
    python predict_url.py https://www.example.com

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import sys
import os
import joblib
import pandas as pd
import argparse

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import URLFeatureExtractor


class PhishingDetector:
    """
    Detector de phishing listo para producción.

    Carga el mejor modelo entrenado y realiza predicciones
    en URLs en tiempo real.
    """

    def __init__(self, models_dir='./models', data_dir='./data/processed'):
        """
        Inicializa el detector.

        Args:
            models_dir (str): Directorio con modelos guardados
            data_dir (str): Directorio con datos procesados
        """
        self.models_dir = models_dir
        self.data_dir = data_dir

        # Cargar componentes
        self._load_components()

    def _load_components(self):
        """Carga modelo, scaler y features."""
        print("Cargando componentes del sistema...")

        # Cargar nombre del mejor modelo
        best_model_path = os.path.join('results', 'metrics', 'best_model.txt')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                content = f.read()
                self.best_model_name = content.split('\n')[0].replace('Best Model: ', '')
        else:
            # Usar un modelo por defecto
            self.best_model_name = 'random_forest'
            print(f"⚠ No se encontró best_model.txt. Usando {self.best_model_name} por defecto.")

        # Cargar modelo
        model_path = os.path.join(self.models_dir, f'{self.best_model_name}.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = joblib.load(model_path)
        print(f"✓ Modelo cargado: {self.best_model_name}")

        # Cargar scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("✓ Scaler cargado")

        # Cargar features seleccionadas
        features_path = os.path.join(self.data_dir, 'selected_features.pkl')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features no encontradas: {features_path}")
        self.selected_features = joblib.load(features_path)
        print(f"✓ Features cargadas: {len(self.selected_features)}")

        # Inicializar extractor
        self.extractor = URLFeatureExtractor()

        print("✓ Sistema listo para predicciones\n")

    def predict(self, url: str, verbose: bool = True) -> dict:
        """
        Predice si una URL es phishing o legítima.

        Args:
            url (str): URL a analizar
            verbose (bool): Mostrar información detallada

        Returns:
            dict: Resultado de la predicción
        """
        # Extraer features
        features = self.extractor.extract_features(url)

        # Convertir a DataFrame
        df_features = pd.DataFrame([features])

        # Seleccionar solo las features usadas en entrenamiento
        X = df_features[self.selected_features]

        # Normalizar
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )

        # Predecir
        prediction = self.model.predict(X_scaled)[0]

        # Obtener probabilidad si está disponible
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[0]
            phishing_prob = probabilities[1]
            legitimate_prob = probabilities[0]
        else:
            phishing_prob = None
            legitimate_prob = None

        # Preparar resultado
        result = {
            'url': url,
            'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
            'is_phishing': bool(prediction == 1),
            'phishing_probability': float(phishing_prob) if phishing_prob is not None else None,
            'legitimate_probability': float(legitimate_prob) if legitimate_prob is not None else None,
            'model': self.best_model_name,
            'risk_level': self._get_risk_level(phishing_prob) if phishing_prob is not None else 'UNKNOWN'
        }

        if verbose:
            self._print_result(result)

        return result

    def _get_risk_level(self, phishing_prob: float) -> str:
        """Determina el nivel de riesgo basado en probabilidad."""
        if phishing_prob >= 0.9:
            return 'VERY HIGH'
        elif phishing_prob >= 0.7:
            return 'HIGH'
        elif phishing_prob >= 0.5:
            return 'MEDIUM'
        elif phishing_prob >= 0.3:
            return 'LOW'
        else:
            return 'VERY LOW'

    def _print_result(self, result: dict):
        """Imprime el resultado de forma legible."""
        print("=" * 70)
        print("                RESULTADO DEL ANÁLISIS")
        print("=" * 70)
        print(f"\n🔍 URL analizada:")
        print(f"   {result['url']}")

        print(f"\n📊 Predicción:")
        if result['is_phishing']:
            print(f"   🚨 {result['prediction']} - ¡POSIBLE AMENAZA!")
        else:
            print(f"   ✅ {result['prediction']} - URL segura")

        if result['phishing_probability'] is not None:
            print(f"\n📈 Probabilidades:")
            print(f"   Phishing:   {result['phishing_probability']:.2%}")
            print(f"   Legítimo:   {result['legitimate_probability']:.2%}")

            print(f"\n⚠️  Nivel de Riesgo: {result['risk_level']}")

        print(f"\n🤖 Modelo usado: {result['model']}")
        print("=" * 70)
        print()

    def predict_batch(self, urls: list, verbose: bool = False) -> list:
        """
        Predice múltiples URLs.

        Args:
            urls (list): Lista de URLs
            verbose (bool): Mostrar información detallada

        Returns:
            list: Lista de resultados
        """
        results = []
        total = len(urls)

        print(f"\nAnalizando {total} URLs...\n")

        for i, url in enumerate(urls, 1):
            print(f"[{i}/{total}] Analizando: {url[:60]}...")
            result = self.predict(url, verbose=False)
            results.append(result)

        # Resumen
        phishing_count = sum(1 for r in results if r['is_phishing'])
        legitimate_count = total - phishing_count

        print("\n" + "=" * 70)
        print("                    RESUMEN")
        print("=" * 70)
        print(f"\nTotal de URLs analizadas: {total}")
        print(f"  ✅ Legítimas: {legitimate_count} ({legitimate_count/total*100:.1f}%)")
        print(f"  🚨 Phishing:  {phishing_count} ({phishing_count/total*100:.1f}%)")
        print("=" * 70)
        print()

        if verbose:
            print("\nResultados detallados:")
            for result in results:
                self._print_result(result)

        return results


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Detector de Phishing - Sistema de ML para análisis de URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python predict_url.py https://www.google.com
  python predict_url.py http://suspicious-site.tk/login.php
  python predict_url.py --batch urls.txt
        """
    )

    parser.add_argument(
        'url',
        nargs='?',
        help='URL a analizar'
    )

    parser.add_argument(
        '--batch',
        '-b',
        type=str,
        help='Archivo de texto con URLs (una por línea)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Mostrar información detallada'
    )

    args = parser.parse_args()

    # Verificar argumentos
    if not args.url and not args.batch:
        parser.print_help()
        print("\n❌ Error: Debe proporcionar una URL o un archivo batch (--batch)")
        sys.exit(1)

    try:
        # Inicializar detector
        detector = PhishingDetector()

        # Modo batch
        if args.batch:
            if not os.path.exists(args.batch):
                print(f"❌ Error: Archivo no encontrado: {args.batch}")
                sys.exit(1)

            with open(args.batch, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]

            results = detector.predict_batch(urls, verbose=args.verbose)

            # Guardar resultados
            output_file = f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            print(f"\n✓ Resultados guardados en: {output_file}")

        # Modo single URL
        else:
            detector.predict(args.url, verbose=True)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
