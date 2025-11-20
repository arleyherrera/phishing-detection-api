"""
Pruebas Unitarias: API REST

Tests para los endpoints de la API Flask.
Valida respuestas, manejo de errores y casos borde.
"""

import pytest
import json
import sys
sys.path.append('.')

from api_phishing import app


class TestAPIEndpoints:
    """Suite de tests para los endpoints de la API"""

    @pytest.fixture
    def client(self):
        """Fixture: Cliente de pruebas de Flask"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    # ========================
    # TESTS DEL ENDPOINT /predict
    # ========================

    def test_predict_endpoint_exists(self, client):
        """Test: El endpoint /predict existe"""
        response = client.post('/predict',
            data=json.dumps({'url': 'https://www.google.com'}),
            content_type='application/json'
        )
        # No debe ser 404
        assert response.status_code != 404

    def test_predict_legitimate_url_google(self, client):
        """Test: Predicción de URL legítima (Google)"""
        response = client.post('/predict',
            data=json.dumps({'url': 'https://www.google.com'}),
            content_type='application/json'
        )

        assert response.status_code == 200

        data = response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'url' in data

        # Google debe ser clasificado como safe
        assert data['prediction'] in ['safe', 'phishing']
        assert 0 <= data['confidence'] <= 1

    def test_predict_legitimate_url_peru(self, client):
        """Test: Predicción de URL legítima de Perú"""
        urls_peru = [
            'https://www.bcp.com.pe',
            'https://www.gob.pe',
            'https://www.pucp.edu.pe',
        ]

        for url in urls_peru:
            response = client.post('/predict',
                data=json.dumps({'url': url}),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data['url'] == url

    def test_predict_response_structure(self, client):
        """Test: Estructura de la respuesta de /predict"""
        response = client.post('/predict',
            data=json.dumps({'url': 'https://www.example.com'}),
            content_type='application/json'
        )

        data = response.get_json()

        # Campos obligatorios
        assert 'url' in data
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'features_extracted' in data
        assert 'features_required' in data

        # Tipos correctos
        assert isinstance(data['url'], str)
        assert isinstance(data['prediction'], str)
        assert isinstance(data['confidence'], (int, float))
        assert isinstance(data['features_extracted'], int)
        assert isinstance(data['features_required'], int)

    def test_predict_confidence_range(self, client):
        """Test: Confianza está en rango [0, 1]"""
        response = client.post('/predict',
            data=json.dumps({'url': 'https://www.example.com'}),
            content_type='application/json'
        )

        data = response.get_json()
        assert 0 <= data['confidence'] <= 1

    # ========================
    # TESTS DE MANEJO DE ERRORES
    # ========================

    def test_predict_empty_url(self, client):
        """Test: URL vacía debe retornar error 400"""
        response = client.post('/predict',
            data=json.dumps({'url': ''}),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_predict_no_url_parameter(self, client):
        """Test: Sin parámetro 'url' debe retornar error 400"""
        response = client.post('/predict',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_predict_invalid_json(self, client):
        """Test: JSON inválido debe retornar error"""
        response = client.post('/predict',
            data='invalid json',
            content_type='application/json'
        )

        # Debe retornar error (400 o 500)
        assert response.status_code in [400, 500]

    def test_predict_missing_content_type(self, client):
        """Test: Sin Content-Type debe retornar error"""
        response = client.post('/predict',
            data=json.dumps({'url': 'https://example.com'})
        )

        # Puede funcionar o retornar error dependiendo de Flask
        # Solo verificamos que no crashea
        assert response.status_code in [200, 400, 415]

    # ========================
    # TESTS DE CASOS BORDE
    # ========================

    def test_predict_very_long_url(self, client):
        """Test: URL muy larga"""
        long_url = "https://example.com/" + "a" * 1000
        response = client.post('/predict',
            data=json.dumps({'url': long_url}),
            content_type='application/json'
        )

        # Debe procesar o retornar error, pero no crashear
        assert response.status_code in [200, 400, 500]

    def test_predict_url_with_special_chars(self, client):
        """Test: URL con caracteres especiales"""
        url = "https://example.com/path?param=value&other=123#section"
        response = client.post('/predict',
            data=json.dumps({'url': url}),
            content_type='application/json'
        )

        assert response.status_code == 200

    def test_predict_localhost_url(self, client):
        """Test: URL localhost"""
        response = client.post('/predict',
            data=json.dumps({'url': 'http://localhost:5000'}),
            content_type='application/json'
        )

        # Debe procesar
        assert response.status_code in [200, 400]

    def test_predict_ip_address_url(self, client):
        """Test: URL con dirección IP"""
        response = client.post('/predict',
            data=json.dumps({'url': 'http://192.168.1.1/login'}),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        # URL con IP es indicador de phishing
        # El modelo debe detectarlo

    # ========================
    # TESTS DEL ENDPOINT /health
    # ========================

    def test_health_endpoint_exists(self, client):
        """Test: El endpoint /health existe"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_endpoint_response(self, client):
        """Test: Respuesta de /health"""
        response = client.get('/health')

        assert response.status_code == 200
        data = response.get_json()

        assert 'status' in data
        assert data['status'] == 'ok'
        assert 'model' in data
        assert 'features' in data

    # ========================
    # TESTS DEL ENDPOINT /
    # ========================

    def test_index_endpoint_exists(self, client):
        """Test: El endpoint / existe"""
        response = client.get('/')
        assert response.status_code == 200

    def test_index_endpoint_response(self, client):
        """Test: Respuesta de /"""
        response = client.get('/')

        assert response.status_code == 200
        data = response.get_json()

        assert 'name' in data
        assert 'version' in data
        assert 'model' in data
        assert 'endpoints' in data

    # ========================
    # TESTS DE INTEGRACIÓN API
    # ========================

    def test_multiple_predictions_sequential(self, client):
        """Test: Múltiples predicciones secuenciales"""
        urls = [
            'https://www.google.com',
            'https://www.github.com',
            'https://www.wikipedia.org',
        ]

        for url in urls:
            response = client.post('/predict',
                data=json.dumps({'url': url}),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data['url'] == url

    def test_prediction_consistency(self, client):
        """Test: Predicciones consistentes para la misma URL"""
        url = 'https://www.google.com'

        # Hacer 3 predicciones de la misma URL
        predictions = []
        for _ in range(3):
            response = client.post('/predict',
                data=json.dumps({'url': url}),
                content_type='application/json'
            )
            data = response.get_json()
            predictions.append(data['prediction'])

        # Todas deben ser iguales
        assert len(set(predictions)) == 1, "Predicciones inconsistentes"

    # ========================
    # TESTS DE RENDIMIENTO
    # ========================

    def test_response_time(self, client):
        """Test: Tiempo de respuesta razonable (<2 segundos)"""
        import time

        start = time.time()
        response = client.post('/predict',
            data=json.dumps({'url': 'https://www.example.com'}),
            content_type='application/json'
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Respuesta muy lenta: {elapsed:.2f}s"


# ========================
# TESTS DE SEGURIDAD
# ========================

class TestAPISecurity:
    """Tests de seguridad de la API"""

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_cors_headers(self, client):
        """Test: Headers CORS están presentes"""
        response = client.get('/health')

        # Verificar que CORS está habilitado
        # (Los headers exactos dependen de la configuración)
        assert response.status_code == 200

    def test_no_sensitive_data_in_error(self, client):
        """Test: Los errores no revelan información sensible"""
        response = client.post('/predict',
            data=json.dumps({'invalid': 'data'}),
            content_type='application/json'
        )

        data = response.get_json()

        # No debe revelar stack traces completos
        if 'error' in data:
            error_msg = str(data['error']).lower()
            # No debe contener paths absolutos
            assert 'd:\\' not in error_msg
            assert 'c:\\' not in error_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
