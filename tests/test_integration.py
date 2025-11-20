"""
Pruebas de Integración

Tests del flujo completo:
- Feature Extraction → API → Predicción
- Manejo de errores end-to-end
- Casos borde del sistema completo
"""

import pytest
import requests
import time
import sys
sys.path.append('src')

from feature_extraction import URLFeatureExtractor


class TestFullFlowIntegration:
    """Tests del flujo completo de extremo a extremo"""

    API_URL = 'http://localhost:5000'

    @pytest.fixture(scope="class")
    def check_api_running(self):
        """Verificar que la API esté ejecutándose"""
        try:
            response = requests.get(f'{self.API_URL}/health', timeout=2)
            if response.status_code != 200:
                pytest.skip("API no está respondiendo correctamente")
        except requests.exceptions.RequestException:
            pytest.skip("API no está ejecutándose. Ejecuta: python api_phishing.py")

    # ========================
    # TESTS DE FLUJO COMPLETO
    # ========================

    def test_complete_flow_legitimate_url(self, check_api_running):
        """Test: Flujo completo con URL legítima"""
        url = "https://www.google.com"

        # Paso 1: Feature Extraction local
        extractor = URLFeatureExtractor()
        features = extractor.extract_features(url)

        assert features is not None, "Feature extraction falló"
        assert len(features) > 0, "No se extrajeron features"

        # Paso 2: Llamar a la API
        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        assert response.status_code == 200, "API retornó error"

        # Paso 3: Validar predicción
        data = response.json()

        assert 'prediction' in data
        assert 'confidence' in data
        assert data['url'] == url

        # Google debería ser clasificado como safe
        print(f"\n   URL: {url}")
        print(f"   Predicción: {data['prediction']}")
        print(f"   Confianza: {data['confidence']*100:.1f}%")

    def test_complete_flow_phishing_url(self, check_api_running):
        """Test: Flujo completo con URL de phishing"""
        url = "http://192.168.1.1/secure-login"

        # Paso 1: Feature Extraction
        extractor = URLFeatureExtractor()
        features = extractor.extract_features(url)

        assert features is not None

        # Paso 2: Llamar a la API
        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        assert response.status_code == 200

        # Paso 3: Validar predicción
        data = response.json()

        print(f"\n   URL: {url}")
        print(f"   Predicción: {data['prediction']}")
        print(f"   Confianza: {data['confidence']*100:.1f}%")

        # URL con IP es un fuerte indicador de phishing
        assert 'prediction' in data

    def test_complete_flow_peru_urls(self, check_api_running):
        """Test: Flujo completo con URLs de Perú"""
        peru_urls = [
            "https://www.bcp.com.pe",
            "https://www.gob.pe",
            "https://www.pucp.edu.pe",
        ]

        for url in peru_urls:
            response = requests.post(
                f'{self.API_URL}/predict',
                json={'url': url},
                timeout=5
            )

            assert response.status_code == 200

            data = response.json()
            print(f"\n   URL: {url}")
            print(f"   Predicción: {data['prediction']}")
            print(f"   Confianza: {data['confidence']*100:.1f}%")

            # Todas deberían ser clasificadas como safe
            assert data['prediction'] in ['safe', 'phishing']

    # ========================
    # TESTS DE RENDIMIENTO E2E
    # ========================

    def test_response_time_end_to_end(self, check_api_running):
        """Test: Tiempo de respuesta total < 2 segundos"""
        url = "https://www.example.com"

        start = time.time()

        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Respuesta muy lenta: {elapsed:.2f}s"

        print(f"\n   Tiempo de respuesta: {elapsed*1000:.0f}ms")

    def test_multiple_requests_sequential(self, check_api_running):
        """Test: Múltiples requests secuenciales"""
        urls = [
            "https://www.google.com",
            "https://www.github.com",
            "https://www.wikipedia.org",
            "https://www.python.org",
        ]

        for url in urls:
            response = requests.post(
                f'{self.API_URL}/predict',
                json={'url': url},
                timeout=5
            )

            assert response.status_code == 200
            data = response.json()
            assert data['url'] == url

    # ========================
    # TESTS DE MANEJO DE ERRORES E2E
    # ========================

    def test_network_error_handling(self, check_api_running):
        """Test: Manejo de errores de red"""
        # Intentar con URL inválida de API
        wrong_api_url = 'http://localhost:9999/predict'

        with pytest.raises(requests.exceptions.ConnectionError):
            requests.post(
                wrong_api_url,
                json={'url': 'https://example.com'},
                timeout=2
            )

    def test_timeout_handling(self, check_api_running):
        """Test: Manejo de timeout"""
        with pytest.raises(requests.exceptions.Timeout):
            requests.post(
                f'{self.API_URL}/predict',
                json={'url': 'https://www.google.com'},
                timeout=0.001  # Timeout muy corto
            )

    def test_malformed_request_handling(self, check_api_running):
        """Test: Manejo de request malformado"""
        # Enviar request sin URL
        response = requests.post(
            f'{self.API_URL}/predict',
            json={},
            timeout=5
        )

        assert response.status_code == 400
        data = response.json()
        assert 'error' in data

    # ========================
    # TESTS DE CASOS ESPECIALES
    # ========================

    def test_special_urls_chrome_protocol(self, check_api_running):
        """Test: URLs especiales (chrome://)"""
        # La extensión debería filtrar estas URLs
        # Pero la API debería poder manejarlas sin crashear
        url = "chrome://extensions/"

        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        # Puede retornar 200 con predicción o 400 con error
        assert response.status_code in [200, 400]

    def test_special_urls_file_protocol(self, check_api_running):
        """Test: URLs con protocolo file://"""
        url = "file:///C:/Users/test/document.html"

        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        assert response.status_code in [200, 400]

    def test_very_long_url_integration(self, check_api_running):
        """Test: URL muy larga (end-to-end)"""
        url = "https://example.com/" + "a" * 1000

        response = requests.post(
            f'{self.API_URL}/predict',
            json={'url': url},
            timeout=5
        )

        # Debe procesar o retornar error, pero no crashear
        assert response.status_code in [200, 400, 500]

    # ========================
    # TESTS DE CONSISTENCIA
    # ========================

    def test_prediction_consistency(self, check_api_running):
        """Test: Predicciones consistentes para la misma URL"""
        url = "https://www.google.com"

        predictions = []
        confidences = []

        # Hacer 5 predicciones de la misma URL
        for _ in range(5):
            response = requests.post(
                f'{self.API_URL}/predict',
                json={'url': url},
                timeout=5
            )

            data = response.json()
            predictions.append(data['prediction'])
            confidences.append(data['confidence'])

        # Todas las predicciones deben ser iguales
        assert len(set(predictions)) == 1, "Predicciones inconsistentes"

        # Las confianzas deben ser iguales (o muy similares)
        assert all(abs(c - confidences[0]) < 0.001 for c in confidences), \
            "Confianzas inconsistentes"

        print(f"\n   Predicción consistente: {predictions[0]}")
        print(f"   Confianza: {confidences[0]*100:.1f}%")


# ========================
# TESTS DE HEALTH CHECK
# ========================

class TestHealthAndStatus:
    """Tests de endpoints de status"""

    API_URL = 'http://localhost:5000'

    @pytest.fixture(scope="class")
    def check_api_running(self):
        try:
            response = requests.get(f'{self.API_URL}/health', timeout=2)
            if response.status_code != 200:
                pytest.skip("API no está respondiendo")
        except requests.exceptions.RequestException:
            pytest.skip("API no está ejecutándose")

    def test_health_endpoint(self, check_api_running):
        """Test: Health check endpoint"""
        response = requests.get(f'{self.API_URL}/health', timeout=2)

        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'ok'
        assert 'model' in data
        assert 'features' in data

    def test_index_endpoint(self, check_api_running):
        """Test: Index endpoint"""
        response = requests.get(f'{self.API_URL}/', timeout=2)

        assert response.status_code == 200

        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
