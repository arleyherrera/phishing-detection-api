"""
feature_extraction.py

Módulo para extracción de características (features) de URLs.

Extrae 56 características de URLs para detección de phishing, basado en el
dataset PhiUSIIL de UCI Machine Learning Repository.

Las características incluyen:
- Características lexicográficas (longitud, caracteres especiales)
- Características de dominio (edad, WHOIS, DNS)
- Características de seguridad (HTTPS, certificados)
- Características de contenido (palabras sospechosas, entropía)

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import re
import logging
from urllib.parse import urlparse, parse_qs
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from datetime import datetime

# Imports opcionales (manejar si no están instalados)
try:
    from tld import get_tld
    TLD_AVAILABLE = True
except ImportError:
    TLD_AVAILABLE = False
    logging.warning("Librería 'tld' no disponible. Algunas features estarán limitadas.")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    logging.warning("Librería 'whois' no disponible. Features de WHOIS estarán limitadas.")

try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False
    logging.warning("Librería 'dnspython' no disponible. Features de DNS estarán limitadas.")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class URLFeatureExtractor:
    """
    Clase para extraer características de URLs para detección de phishing.

    Implementa 56 features basadas en investigación de seguridad web y
    el dataset PhiUSIIL.
    """

    def __init__(self):
        """Inicializa el extractor de features."""
        # Palabras sospechosas comunes en phishing
        self.suspicious_words = [
            'account', 'update', 'secure', 'verify', 'login', 'signin',
            'banking', 'confirm', 'suspended', 'unusual', 'click',
            'verify', 'password', 'credential', 'alert', 'notification'
        ]

        # TLDs sospechosos
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']

        logger.info("URLFeatureExtractor inicializado")

    # ========================================================================
    # FEATURES LEXICOGRÁFICAS (Basadas en la estructura de la URL)
    # ========================================================================

    def extract_url_length(self, url: str) -> int:
        """Feature 1: Longitud total de la URL."""
        return len(url)

    def extract_domain_length(self, url: str) -> int:
        """Feature 2: Longitud del dominio."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return len(domain)
        except:
            return 0

    def extract_path_length(self, url: str) -> int:
        """Feature 3: Longitud del path."""
        try:
            parsed = urlparse(url)
            return len(parsed.path)
        except:
            return 0

    def count_dots(self, url: str) -> int:
        """Feature 4: Cantidad de puntos en la URL."""
        return url.count('.')

    def count_hyphens(self, url: str) -> int:
        """Feature 5: Cantidad de guiones en la URL."""
        return url.count('-')

    def count_underscores(self, url: str) -> int:
        """Feature 6: Cantidad de guiones bajos."""
        return url.count('_')

    def count_slashes(self, url: str) -> int:
        """Feature 7: Cantidad de barras."""
        return url.count('/')

    def count_question_marks(self, url: str) -> int:
        """Feature 8: Cantidad de signos de interrogación."""
        return url.count('?')

    def count_equal_signs(self, url: str) -> int:
        """Feature 9: Cantidad de signos igual."""
        return url.count('=')

    def count_at_signs(self, url: str) -> int:
        """Feature 10: Cantidad de @ (sospechoso en URLs)."""
        return url.count('@')

    def count_ampersands(self, url: str) -> int:
        """Feature 11: Cantidad de &."""
        return url.count('&')

    def count_digits(self, url: str) -> int:
        """Feature 12: Cantidad de dígitos en la URL."""
        return sum(c.isdigit() for c in url)

    def count_letters(self, url: str) -> int:
        """Feature 13: Cantidad de letras."""
        return sum(c.isalpha() for c in url)

    def count_special_chars(self, url: str) -> int:
        """Feature 14: Cantidad de caracteres especiales."""
        special = set('!#$%^&*()_+-=[]{}|;:,.<>?`~')
        return sum(c in special for c in url)

    # ========================================================================
    # FEATURES DE DOMINIO
    # ========================================================================

    def has_ip_address(self, url: str) -> int:
        """Feature 15: ¿Usa dirección IP en lugar de dominio? (1=Sí, 0=No)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]  # Remover puerto si existe
            # Regex para IPv4
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            return 1 if re.match(ip_pattern, domain) else 0
        except:
            return 0

    def count_subdomains(self, url: str) -> int:
        """Feature 16: Cantidad de subdominios."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Contar puntos en el dominio (subdominios)
            return domain.count('.')
        except:
            return 0

    def domain_has_hyphen(self, url: str) -> int:
        """Feature 17: ¿El dominio contiene guión? (1=Sí, 0=No)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return 1 if '-' in domain else 0
        except:
            return 0

    def has_suspicious_tld(self, url: str) -> int:
        """Feature 18: ¿Usa TLD sospechoso? (1=Sí, 0=No)."""
        try:
            for tld in self.suspicious_tlds:
                if url.lower().endswith(tld):
                    return 1
            return 0
        except:
            return 0

    # ========================================================================
    # FEATURES DE PROTOCOLO Y SEGURIDAD
    # ========================================================================

    def is_https(self, url: str) -> int:
        """Feature 19: ¿Usa HTTPS? (1=Sí, 0=No)."""
        return 1 if url.lower().startswith('https://') else 0

    def has_port(self, url: str) -> int:
        """Feature 20: ¿Especifica puerto en la URL? (1=Sí, 0=No)."""
        try:
            parsed = urlparse(url)
            return 1 if parsed.port is not None else 0
        except:
            return 0

    # ========================================================================
    # FEATURES DE PATH Y QUERY
    # ========================================================================

    def count_path_segments(self, url: str) -> int:
        """Feature 21: Cantidad de segmentos en el path."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            return len(path.split('/')) if path else 0
        except:
            return 0

    def count_query_parameters(self, url: str) -> int:
        """Feature 22: Cantidad de parámetros en query string."""
        try:
            parsed = urlparse(url)
            return len(parse_qs(parsed.query))
        except:
            return 0

    def has_fragment(self, url: str) -> int:
        """Feature 23: ¿Tiene fragmento (#)? (1=Sí, 0=No)."""
        try:
            parsed = urlparse(url)
            return 1 if parsed.fragment else 0
        except:
            return 0

    # ========================================================================
    # FEATURES DE CONTENIDO Y PALABRAS SOSPECHOSAS
    # ========================================================================

    def count_suspicious_words(self, url: str) -> int:
        """Feature 24: Cantidad de palabras sospechosas en la URL."""
        url_lower = url.lower()
        return sum(word in url_lower for word in self.suspicious_words)

    def has_login_word(self, url: str) -> int:
        """Feature 25: ¿Contiene 'login' o 'signin'? (1=Sí, 0=No)."""
        url_lower = url.lower()
        return 1 if ('login' in url_lower or 'signin' in url_lower) else 0

    def has_verify_word(self, url: str) -> int:
        """Feature 26: ¿Contiene 'verify' o 'confirm'? (1=Sí, 0=No)."""
        url_lower = url.lower()
        return 1 if ('verify' in url_lower or 'confirm' in url_lower) else 0

    # ========================================================================
    # FEATURES DE ENTROPÍA Y ALEATORIEDAD
    # ========================================================================

    def calculate_entropy(self, text: str) -> float:
        """
        Feature 27: Entropía de Shannon de la URL.
        Mayor entropía = más aleatoriedad = más sospechoso.
        """
        if not text:
            return 0.0

        # Calcular frecuencia de cada carácter
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # Calcular entropía
        entropy = 0.0
        text_len = len(text)
        for count in freq.values():
            prob = count / text_len
            entropy -= prob * math.log2(prob)

        return entropy

    def domain_entropy(self, url: str) -> float:
        """Feature 28: Entropía del dominio."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return self.calculate_entropy(domain)
        except:
            return 0.0

    def path_entropy(self, url: str) -> float:
        """Feature 29: Entropía del path."""
        try:
            parsed = urlparse(url)
            return self.calculate_entropy(parsed.path)
        except:
            return 0.0

    # ========================================================================
    # FEATURES DE RATIOS
    # ========================================================================

    def digit_letter_ratio(self, url: str) -> float:
        """Feature 30: Ratio de dígitos respecto a letras."""
        digits = self.count_digits(url)
        letters = self.count_letters(url)
        return digits / letters if letters > 0 else 0.0

    def special_char_ratio(self, url: str) -> float:
        """Feature 31: Ratio de caracteres especiales respecto al total."""
        special = self.count_special_chars(url)
        total = len(url)
        return special / total if total > 0 else 0.0

    # ========================================================================
    # FEATURES ADICIONALES (32-56)
    # ========================================================================

    def has_double_slash_redirect(self, url: str) -> int:
        """Feature 32: ¿Tiene // fuera del protocolo? (técnica de redirect)."""
        url_without_protocol = url.split('://', 1)[-1]
        return 1 if '//' in url_without_protocol else 0

    def abnormal_url_length(self, url: str) -> int:
        """Feature 33: ¿URL anormalmente larga? (>75 caracteres)."""
        return 1 if len(url) > 75 else 0

    def tiny_url_length(self, url: str) -> int:
        """Feature 34: ¿URL muy corta? (<54 caracteres)."""
        return 1 if len(url) < 54 else 0

    def prefix_suffix_domain(self, url: str) -> int:
        """Feature 35: ¿Dominio tiene prefijo-sufijo? (ej: paypal-secure.com)."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return 1 if '-' in domain else 0
        except:
            return 0

    def count_www(self, url: str) -> int:
        """Feature 36: Cantidad de veces que aparece 'www'."""
        return url.lower().count('www')

    def count_com(self, url: str) -> int:
        """Feature 37: Cantidad de veces que aparece 'com'."""
        return url.lower().count('com')

    def count_double_slash(self, url: str) -> int:
        """Feature 38: Cantidad de '//'."""
        return url.count('//')

    def count_https_token(self, url: str) -> int:
        """Feature 39: Cantidad de veces 'https' (excluyendo protocolo)."""
        url_without_protocol = url.split('://', 1)[-1]
        return url_without_protocol.lower().count('https')

    def count_http_token(self, url: str) -> int:
        """Feature 40: Cantidad de veces 'http' en dominio/path."""
        url_without_protocol = url.split('://', 1)[-1]
        return url_without_protocol.lower().count('http')

    def ratio_digits_url(self, url: str) -> float:
        """Feature 41: Ratio de dígitos en la URL."""
        return self.count_digits(url) / len(url) if len(url) > 0 else 0.0

    def ratio_digits_domain(self, url: str) -> float:
        """Feature 42: Ratio de dígitos en el dominio."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if len(domain) > 0:
                digits = sum(c.isdigit() for c in domain)
                return digits / len(domain)
            return 0.0
        except:
            return 0.0

    def punycode_domain(self, url: str) -> int:
        """Feature 43: ¿Usa punycode? (xn--)."""
        return 1 if 'xn--' in url.lower() else 0

    def count_percentage(self, url: str) -> int:
        """Feature 44: Cantidad de símbolos %."""
        return url.count('%')

    def count_tilde(self, url: str) -> int:
        """Feature 45: Cantidad de ~."""
        return url.count('~')

    def count_asterisk(self, url: str) -> int:
        """Feature 46: Cantidad de *."""
        return url.count('*')

    def count_colon(self, url: str) -> int:
        """Feature 47: Cantidad de :."""
        return url.count(':')

    def count_semicolon(self, url: str) -> int:
        """Feature 48: Cantidad de ;."""
        return url.count(';')

    def count_dollar(self, url: str) -> int:
        """Feature 49: Cantidad de $."""
        return url.count('$')

    def count_space(self, url: str) -> int:
        """Feature 50: Cantidad de espacios (URL encoding)."""
        return url.count(' ') + url.count('%20')

    def avg_token_length_domain(self, url: str) -> float:
        """Feature 51: Longitud promedio de tokens en dominio."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            tokens = domain.split('.')
            if len(tokens) > 0:
                avg_len = sum(len(t) for t in tokens) / len(tokens)
                return avg_len
            return 0.0
        except:
            return 0.0

    def avg_token_length_path(self, url: str) -> float:
        """Feature 52: Longitud promedio de tokens en path."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            if path:
                tokens = path.split('/')
                avg_len = sum(len(t) for t in tokens) / len(tokens)
                return avg_len
            return 0.0
        except:
            return 0.0

    def longest_token_length(self, url: str) -> int:
        """Feature 53: Longitud del token más largo."""
        tokens = re.split(r'[./\-_?=&@]', url)
        return max(len(t) for t in tokens) if tokens else 0

    def has_client_in_url(self, url: str) -> int:
        """Feature 54: ¿Contiene 'client'?."""
        return 1 if 'client' in url.lower() else 0

    def has_admin_in_url(self, url: str) -> int:
        """Feature 55: ¿Contiene 'admin'?."""
        return 1 if 'admin' in url.lower() else 0

    def has_server_in_url(self, url: str) -> int:
        """Feature 56: ¿Contiene 'server'?."""
        return 1 if 'server' in url.lower() else 0

    # ========================================================================
    # FUNCIÓN PRINCIPAL DE EXTRACCIÓN
    # ========================================================================

    def extract_features(self, url: str) -> Dict[str, float]:
        """
        Extrae todas las 56 características de una URL.

        Args:
            url (str): URL a analizar

        Returns:
            Dict[str, float]: Diccionario con todas las features
        """
        features = {
            # Lexicográficas (1-14)
            'url_length': self.extract_url_length(url),
            'domain_length': self.extract_domain_length(url),
            'path_length': self.extract_path_length(url),
            'count_dots': self.count_dots(url),
            'count_hyphens': self.count_hyphens(url),
            'count_underscores': self.count_underscores(url),
            'count_slashes': self.count_slashes(url),
            'count_question_marks': self.count_question_marks(url),
            'count_equal_signs': self.count_equal_signs(url),
            'count_at_signs': self.count_at_signs(url),
            'count_ampersands': self.count_ampersands(url),
            'count_digits': self.count_digits(url),
            'count_letters': self.count_letters(url),
            'count_special_chars': self.count_special_chars(url),

            # Dominio (15-18)
            'has_ip_address': self.has_ip_address(url),
            'count_subdomains': self.count_subdomains(url),
            'domain_has_hyphen': self.domain_has_hyphen(url),
            'has_suspicious_tld': self.has_suspicious_tld(url),

            # Protocolo y seguridad (19-20)
            'is_https': self.is_https(url),
            'has_port': self.has_port(url),

            # Path y query (21-23)
            'count_path_segments': self.count_path_segments(url),
            'count_query_parameters': self.count_query_parameters(url),
            'has_fragment': self.has_fragment(url),

            # Palabras sospechosas (24-26)
            'count_suspicious_words': self.count_suspicious_words(url),
            'has_login_word': self.has_login_word(url),
            'has_verify_word': self.has_verify_word(url),

            # Entropía (27-29)
            'url_entropy': self.calculate_entropy(url),
            'domain_entropy': self.domain_entropy(url),
            'path_entropy': self.path_entropy(url),

            # Ratios (30-31)
            'digit_letter_ratio': self.digit_letter_ratio(url),
            'special_char_ratio': self.special_char_ratio(url),

            # Adicionales (32-56)
            'has_double_slash_redirect': self.has_double_slash_redirect(url),
            'abnormal_url_length': self.abnormal_url_length(url),
            'tiny_url_length': self.tiny_url_length(url),
            'prefix_suffix_domain': self.prefix_suffix_domain(url),
            'count_www': self.count_www(url),
            'count_com': self.count_com(url),
            'count_double_slash': self.count_double_slash(url),
            'count_https_token': self.count_https_token(url),
            'count_http_token': self.count_http_token(url),
            'ratio_digits_url': self.ratio_digits_url(url),
            'ratio_digits_domain': self.ratio_digits_domain(url),
            'punycode_domain': self.punycode_domain(url),
            'count_percentage': self.count_percentage(url),
            'count_tilde': self.count_tilde(url),
            'count_asterisk': self.count_asterisk(url),
            'count_colon': self.count_colon(url),
            'count_semicolon': self.count_semicolon(url),
            'count_dollar': self.count_dollar(url),
            'count_space': self.count_space(url),
            'avg_token_length_domain': self.avg_token_length_domain(url),
            'avg_token_length_path': self.avg_token_length_path(url),
            'longest_token_length': self.longest_token_length(url),
            'has_client_in_url': self.has_client_in_url(url),
            'has_admin_in_url': self.has_admin_in_url(url),
            'has_server_in_url': self.has_server_in_url(url),
        }

        return features

    def extract_features_batch(self, urls: List[str], show_progress: bool = True) -> pd.DataFrame:
        """
        Extrae features de múltiples URLs en batch.

        Args:
            urls (List[str]): Lista de URLs
            show_progress (bool): Mostrar barra de progreso

        Returns:
            pd.DataFrame: DataFrame con features de todas las URLs
        """
        logger.info(f"Extrayendo features de {len(urls)} URLs...")

        features_list = []

        iterator = tqdm(urls, desc="Extrayendo features") if show_progress else urls

        for url in iterator:
            try:
                features = self.extract_features(url)
                features['url'] = url
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error extrayendo features de {url}: {e}")
                # Agregar features vacías
                features_list.append({'url': url})

        df = pd.DataFrame(features_list)
        logger.info(f"Features extraídas: {df.shape}")

        return df


def main():
    """
    Función principal para probar el extractor de features.

    Ejemplo de uso:
        python feature_extraction.py
    """
    # Ejemplos de URLs para probar
    test_urls = [
        'https://www.google.com',
        'http://secure-login-paypal.tk/verify.php?id=123',
        'https://www.bcp.com.pe',
        'http://192.168.1.1/admin/login.php'
    ]

    extractor = URLFeatureExtractor()

    print("=== Extrayendo features de URLs de prueba ===\n")

    for url in test_urls:
        print(f"URL: {url}")
        features = extractor.extract_features(url)
        print(f"  - Longitud: {features['url_length']}")
        print(f"  - HTTPS: {features['is_https']}")
        print(f"  - Palabras sospechosas: {features['count_suspicious_words']}")
        print(f"  - Entropía: {features['url_entropy']:.2f}")
        print()

    # Batch extraction
    print("=== Extracción en batch ===")
    df_features = extractor.extract_features_batch(test_urls)
    print(df_features.head())


if __name__ == "__main__":
    main()
