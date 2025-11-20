"""
Pruebas Unitarias: Feature Extraction

Tests para el extractor de características de URLs.
Valida que las features se extraen correctamente.
"""

import pytest
import sys
sys.path.append('src')

from feature_extraction import URLFeatureExtractor


class TestURLFeatureExtractor:
    """Suite de tests para URLFeatureExtractor"""

    @pytest.fixture
    def extractor(self):
        """Fixture: Crear instancia del extractor"""
        return URLFeatureExtractor()

    # ========================
    # TESTS BÁSICOS
    # ========================

    def test_extract_basic_features_google(self, extractor):
        """Test: Extraer features básicas de google.com"""
        url = "https://www.google.com"
        features = extractor.extract_features(url)

        # Validar que se extrajeron features
        assert features is not None
        assert len(features) > 0

        # Validar features específicas
        assert 'url_length' in features
        assert 'domain_length' in features
        assert 'is_https' in features

        # Validar valores esperados
        assert features['is_https'] == 1  # Google usa HTTPS
        assert features['has_www'] == 1   # tiene www

    def test_extract_features_github(self, extractor):
        """Test: Extraer features de github.com"""
        url = "https://github.com/anthropics/claude-code"
        features = extractor.extract_features(url)

        assert features is not None
        assert features['is_https'] == 1
        assert features['url_length'] > 30
        assert features['count_slashes'] >= 2

    # ========================
    # TESTS DE PHISHING INDICATORS
    # ========================

    def test_detect_ip_in_url(self, extractor):
        """Test: Detectar IP en URL (indicador de phishing)"""
        url = "http://192.168.1.1/login"
        features = extractor.extract_features(url)

        assert features is not None
        assert 'has_ip' in features
        assert features['has_ip'] == 1  # Debe detectar la IP

    def test_detect_suspicious_words(self, extractor):
        """Test: Detectar palabras sospechosas"""
        url = "http://secure-login-account-verify.com"
        features = extractor.extract_features(url)

        assert features is not None
        if 'has_suspicious_words' in features:
            # Debe detectar palabras como "login", "secure", "account"
            assert features['has_suspicious_words'] == 1

    def test_detect_many_subdomains(self, extractor):
        """Test: Detectar múltiples subdominios (sospechoso)"""
        url = "http://www.secure.login.phishing.example.com"
        features = extractor.extract_features(url)

        assert features is not None
        if 'num_subdomains' in features:
            assert features['num_subdomains'] >= 4

    def test_detect_long_url(self, extractor):
        """Test: Detectar URL muy larga (sospechoso)"""
        url = "http://example.com/" + "a" * 200
        features = extractor.extract_features(url)

        assert features is not None
        assert features['url_length'] > 200

    def test_detect_special_characters(self, extractor):
        """Test: Contar caracteres especiales"""
        url = "http://test@example.com/path?param=value&other=123"
        features = extractor.extract_features(url)

        assert features is not None
        assert features['count_at_signs'] >= 1
        assert features['count_question_marks'] >= 1
        assert features['count_equal_signs'] >= 2

    # ========================
    # TESTS DE CASOS BORDE
    # ========================

    def test_empty_url(self, extractor):
        """Test: URL vacía"""
        features = extractor.extract_features("")

        # Debe retornar None o dict vacío
        assert features is None or len(features) == 0

    def test_malformed_url_no_protocol(self, extractor):
        """Test: URL sin protocolo"""
        url = "www.example.com"
        features = extractor.extract_features(url)

        # Debe intentar extraer features básicas
        assert features is not None

    def test_url_with_port(self, extractor):
        """Test: URL con puerto"""
        url = "http://example.com:8080/path"
        features = extractor.extract_features(url)

        assert features is not None
        assert features['url_length'] > 0

    def test_url_with_fragment(self, extractor):
        """Test: URL con fragmento (#)"""
        url = "https://example.com/page#section"
        features = extractor.extract_features(url)

        assert features is not None

    def test_localhost_url(self, extractor):
        """Test: URL localhost"""
        url = "http://localhost:5000/predict"
        features = extractor.extract_features(url)

        assert features is not None
        assert features['url_length'] > 0

    # ========================
    # TESTS DE FEATURES ESPECÍFICAS
    # ========================

    def test_https_detection(self, extractor):
        """Test: Detección correcta de HTTPS"""
        url_https = "https://secure.example.com"
        url_http = "http://insecure.example.com"

        features_https = extractor.extract_features(url_https)
        features_http = extractor.extract_features(url_http)

        assert features_https['is_https'] == 1
        assert features_http['is_https'] == 0

    def test_www_detection(self, extractor):
        """Test: Detección de www"""
        url_with_www = "https://www.example.com"
        url_without_www = "https://example.com"

        features_with = extractor.extract_features(url_with_www)
        features_without = extractor.extract_features(url_without_www)

        if 'has_www' in features_with:
            assert features_with['has_www'] == 1
            assert features_without['has_www'] == 0

    def test_url_length_calculation(self, extractor):
        """Test: Cálculo correcto de longitud de URL"""
        url = "https://example.com"
        features = extractor.extract_features(url)

        assert features['url_length'] == len(url)

    def test_domain_length_calculation(self, extractor):
        """Test: Cálculo correcto de longitud de dominio"""
        url = "https://example.com/path"
        features = extractor.extract_features(url)

        # Dominio es "example.com" (11 caracteres)
        assert features['domain_length'] == len("example.com")

    # ========================
    # TESTS DE RENDIMIENTO
    # ========================

    def test_extraction_speed(self, extractor):
        """Test: Velocidad de extracción (debe ser < 1 segundo)"""
        import time

        url = "https://www.example.com/test/path"

        start = time.time()
        features = extractor.extract_features(url)
        elapsed = time.time() - start

        assert features is not None
        assert elapsed < 1.0, f"Extracción muy lenta: {elapsed:.2f}s"

    def test_extract_multiple_urls(self, extractor):
        """Test: Extraer features de múltiples URLs"""
        urls = [
            "https://www.google.com",
            "https://www.github.com",
            "http://example.com",
            "https://www.wikipedia.org",
        ]

        for url in urls:
            features = extractor.extract_features(url)
            assert features is not None
            assert len(features) > 0


# ========================
# TESTS DE CASOS REALES
# ========================

class TestRealURLs:
    """Tests con URLs reales de Perú"""

    @pytest.fixture
    def extractor(self):
        return URLFeatureExtractor()

    def test_peru_legitimate_urls(self, extractor):
        """Test: URLs legítimas de Perú"""
        legitimate_urls = [
            "https://www.bcp.com.pe",
            "https://www.gob.pe",
            "https://www.pucp.edu.pe",
            "https://www.sunat.gob.pe",
        ]

        for url in legitimate_urls:
            features = extractor.extract_features(url)

            # Validaciones básicas
            assert features is not None
            assert features['is_https'] == 1  # Todas usan HTTPS

    def test_phishing_urls(self, extractor):
        """Test: URLs de phishing conocidas"""
        phishing_urls = [
            "http://192.168.1.1/secure-login",
            "http://verify-account-login.xyz",
        ]

        for url in phishing_urls:
            features = extractor.extract_features(url)
            assert features is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
