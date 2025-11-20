"""
data_collection.py

Módulo para recolección de datos de phishing y URLs legítimas.

Este módulo proporciona funcionalidades para:
- Descargar datos de PhishTank (URLs de phishing verificadas)
- Recolectar URLs legítimas de sitios web peruanos
- Validar y limpiar URLs recolectadas
- Guardar datasets en formato estructurado

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Tuple
from urllib.parse import urlparse
import time
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishTankCollector:
    """
    Clase para recolectar datos de PhishTank.

    PhishTank es una comunidad que recopila y verifica URLs de phishing.
    Proporciona una base de datos actualizada de sitios fraudulentos.
    """

    def __init__(self, data_dir: str = "../data/external"):
        """
        Inicializa el recolector de PhishTank.

        Args:
            data_dir (str): Directorio donde se guardarán los datos descargados
        """
        self.data_dir = data_dir
        self.phishtank_url = "http://data.phishtank.com/data/online-valid.csv"
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"PhishTankCollector inicializado. Directorio: {data_dir}")

    def download_phishtank_data(self) -> pd.DataFrame:
        """
        Descarga el dataset actualizado de PhishTank.

        Returns:
            pd.DataFrame: DataFrame con URLs de phishing y metadata

        Raises:
            requests.exceptions.RequestException: Si falla la descarga
        """
        logger.info("Descargando datos de PhishTank...")

        try:
            # Descargar archivo CSV de PhishTank
            response = requests.get(self.phishtank_url, timeout=60)
            response.raise_for_status()

            # Guardar archivo localmente
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.data_dir, f"phishtank_{timestamp}.csv")

            with open(filepath, 'wb') as f:
                f.write(response.content)

            logger.info(f"Archivo descargado: {filepath}")

            # Leer CSV
            df = pd.read_csv(filepath)
            logger.info(f"URLs de phishing descargadas: {len(df)}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error al descargar datos de PhishTank: {e}")
            raise

    def clean_phishtank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y procesa los datos de PhishTank.

        Args:
            df (pd.DataFrame): DataFrame crudo de PhishTank

        Returns:
            pd.DataFrame: DataFrame limpio con columnas relevantes
        """
        logger.info("Limpiando datos de PhishTank...")

        # Seleccionar columnas relevantes (PhishTank tiene: phish_id, url, phish_detail_url, etc.)
        if 'url' in df.columns:
            df_clean = df[['url']].copy()
            df_clean['label'] = 1  # 1 = phishing
            df_clean['source'] = 'PhishTank'
            df_clean['collection_date'] = datetime.now().strftime("%Y-%m-%d")

            # Eliminar duplicados
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['url'])
            logger.info(f"Duplicados eliminados: {initial_count - len(df_clean)}")

            # Eliminar URLs vacías o inválidas
            df_clean = df_clean[df_clean['url'].notna()]
            df_clean = df_clean[df_clean['url'].str.strip() != '']

            logger.info(f"URLs de phishing limpias: {len(df_clean)}")
            return df_clean
        else:
            logger.error("Columna 'url' no encontrada en datos de PhishTank")
            raise KeyError("Columna 'url' no encontrada")


class PeruLegitimateURLCollector:
    """
    Clase para recolectar URLs legítimas de sitios web peruanos.

    Recopila URLs de instituciones confiables de Perú en diversas categorías:
    - Bancos
    - Gobierno
    - Universidades
    - E-commerce
    - Medios de comunicación
    """

    def __init__(self, data_dir: str = "../data/external"):
        """
        Inicializa el recolector de URLs peruanas legítimas.

        Args:
            data_dir (str): Directorio donde se guardarán los datos
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Definir URLs legítimas por categoría
        self.legitimate_urls = {
            'bancos': [
                'https://www.viabcp.com',
                'https://www.bcp.com.pe',
                'https://interbank.pe',
                'https://www.bbva.pe',
                'https://www.scotiabank.com.pe',
                'https://www.banbif.com.pe',
                'https://www.bancodelnacion.gob.pe',
                'https://www.pichincha.pe',
                'https://www.bancofalabella.pe',
                'https://www.mibanco.com.pe',
            ],
            'gobierno': [
                'https://www.gob.pe',
                'https://www.sunat.gob.pe',
                'https://www.reniec.gob.pe',
                'https://www.minsa.gob.pe',
                'https://www.minedu.gob.pe',
                'https://www.pcm.gob.pe',
                'https://www.mef.gob.pe',
                'https://www.mtc.gob.pe',
                'https://www.produce.gob.pe',
                'https://www.mininter.gob.pe',
                'https://www.mindef.gob.pe',
                'https://www.midis.gob.pe',
            ],
            'universidades': [
                'https://www.pucp.edu.pe',
                'https://www.uni.edu.pe',
                'https://www.unmsm.edu.pe',
                'https://www.usmp.edu.pe',
                'https://www.upc.edu.pe',
                'https://www.ulima.edu.pe',
                'https://www.esan.edu.pe',
                'https://www.up.edu.pe',
                'https://www.utec.edu.pe',
                'https://www.ucsur.edu.pe',
                'https://www.unfv.edu.pe',
                'https://www.upn.edu.pe',
            ],
            'ecommerce': [
                'https://www.falabella.com.pe',
                'https://www.ripley.com.pe',
                'https://www.plazavea.com.pe',
                'https://www.tottus.com.pe',
                'https://www.metro.pe',
                'https://www.wong.pe',
                'https://www.sodimac.com.pe',
                'https://www.linio.com.pe',
                'https://www.mercadolibre.com.pe',
                'https://www.juntoz.com',
            ],
            'medios': [
                'https://www.elcomercio.pe',
                'https://www.rpp.pe',
                'https://www.larepublica.pe',
                'https://www.peru21.pe',
                'https://www.gestion.pe',
                'https://www.trome.pe',
                'https://www.depor.com',
                'https://www.americatv.com.pe',
                'https://www.panamericana.pe',
                'https://www.willax.tv',
            ],
            'servicios': [
                'https://www.sedapal.com.pe',
                'https://www.luz-del-sur.com.pe',
                'https://www.movistar.com.pe',
                'https://www.claro.com.pe',
                'https://www.entel.pe',
                'https://www.essalud.gob.pe',
                'https://www.sbs.gob.pe',
                'https://www.indecopi.gob.pe',
            ]
        }

        logger.info(f"PeruLegitimateURLCollector inicializado. Directorio: {data_dir}")

    def collect_legitimate_urls(self) -> pd.DataFrame:
        """
        Recopila todas las URLs legítimas peruanas.

        Returns:
            pd.DataFrame: DataFrame con URLs legítimas y metadata
        """
        logger.info("Recopilando URLs legítimas de Perú...")

        urls_data = []

        for category, urls in self.legitimate_urls.items():
            for url in urls:
                urls_data.append({
                    'url': url,
                    'label': 0,  # 0 = legítimo
                    'source': f'Peru_{category}',
                    'category': category,
                    'collection_date': datetime.now().strftime("%Y-%m-%d")
                })

        df = pd.DataFrame(urls_data)
        logger.info(f"URLs legítimas recopiladas: {len(df)}")
        logger.info(f"Categorías: {df['category'].value_counts().to_dict()}")

        return df

    def validate_urls(self, df: pd.DataFrame, timeout: int = 5) -> pd.DataFrame:
        """
        Valida que las URLs sean accesibles (opcional).

        Args:
            df (pd.DataFrame): DataFrame con URLs
            timeout (int): Timeout para cada request en segundos

        Returns:
            pd.DataFrame: DataFrame con URLs validadas
        """
        logger.info("Validando URLs (verificando accesibilidad)...")

        valid_urls = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validando URLs"):
            try:
                response = requests.head(row['url'], timeout=timeout, allow_redirects=True)
                if response.status_code < 400:
                    valid_urls.append(row)
                else:
                    logger.warning(f"URL no accesible (HTTP {response.status_code}): {row['url']}")
            except Exception as e:
                logger.warning(f"Error validando URL {row['url']}: {e}")

            # Pequeña pausa para no saturar los servidores
            time.sleep(0.1)

        df_valid = pd.DataFrame(valid_urls)
        logger.info(f"URLs validadas: {len(df_valid)} de {len(df)}")

        return df_valid


class DataCollectionManager:
    """
    Clase principal para gestionar toda la recolección de datos.

    Coordina la recolección de datos de PhishTank y URLs peruanas,
    combina los datasets y los guarda en formato adecuado.
    """

    def __init__(self, data_dir: str = "../data/external"):
        """
        Inicializa el gestor de recolección de datos.

        Args:
            data_dir (str): Directorio donde se guardarán los datos
        """
        self.data_dir = data_dir
        self.phishtank_collector = PhishTankCollector(data_dir)
        self.peru_collector = PeruLegitimateURLCollector(data_dir)
        logger.info("DataCollectionManager inicializado")

    def collect_all_data(self, validate_peru_urls: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Recolecta todos los datos: phishing de PhishTank y URLs legítimas de Perú.

        Args:
            validate_peru_urls (bool): Si True, valida la accesibilidad de URLs peruanas

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_phishing, df_legitimate)
        """
        logger.info("=== Iniciando recolección completa de datos ===")

        # 1. Recolectar datos de PhishTank
        try:
            df_phishtank_raw = self.phishtank_collector.download_phishtank_data()
            df_phishing = self.phishtank_collector.clean_phishtank_data(df_phishtank_raw)
        except Exception as e:
            logger.error(f"Error recolectando datos de PhishTank: {e}")
            logger.warning("Continuando sin datos de PhishTank...")
            df_phishing = pd.DataFrame(columns=['url', 'label', 'source', 'collection_date'])

        # 2. Recolectar URLs legítimas de Perú
        df_legitimate = self.peru_collector.collect_legitimate_urls()

        if validate_peru_urls:
            df_legitimate = self.peru_collector.validate_urls(df_legitimate)

        logger.info(f"=== Recolección completada ===")
        logger.info(f"Total URLs de phishing: {len(df_phishing)}")
        logger.info(f"Total URLs legítimas: {len(df_legitimate)}")

        return df_phishing, df_legitimate

    def save_collected_data(self, df_phishing: pd.DataFrame, df_legitimate: pd.DataFrame) -> str:
        """
        Guarda los datos recolectados en archivos CSV.

        Args:
            df_phishing (pd.DataFrame): DataFrame de URLs de phishing
            df_legitimate (pd.DataFrame): DataFrame de URLs legítimas

        Returns:
            str: Path al archivo combinado guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar por separado
        phishing_path = os.path.join(self.data_dir, f"phishing_urls_{timestamp}.csv")
        legitimate_path = os.path.join(self.data_dir, f"legitimate_urls_{timestamp}.csv")

        df_phishing.to_csv(phishing_path, index=False)
        df_legitimate.to_csv(legitimate_path, index=False)

        logger.info(f"URLs de phishing guardadas: {phishing_path}")
        logger.info(f"URLs legítimas guardadas: {legitimate_path}")

        # Combinar y guardar dataset completo
        df_combined = pd.concat([df_phishing, df_legitimate], ignore_index=True)
        combined_path = os.path.join(self.data_dir, f"combined_urls_{timestamp}.csv")
        df_combined.to_csv(combined_path, index=False)

        logger.info(f"Dataset combinado guardado: {combined_path}")
        logger.info(f"Total URLs en dataset combinado: {len(df_combined)}")
        logger.info(f"Distribución de clases:\n{df_combined['label'].value_counts()}")

        return combined_path


def main():
    """
    Función principal para ejecutar la recolección de datos.

    Ejemplo de uso:
        python data_collection.py
    """
    logger.info("=== Iniciando proceso de recolección de datos ===")

    # Crear directorio si no existe
    data_dir = "../data/external"
    os.makedirs(data_dir, exist_ok=True)

    # Inicializar manager
    manager = DataCollectionManager(data_dir)

    # Recolectar datos (validate_peru_urls=False para ser más rápido)
    df_phishing, df_legitimate = manager.collect_all_data(validate_peru_urls=False)

    # Guardar datos
    combined_path = manager.save_collected_data(df_phishing, df_legitimate)

    logger.info(f"=== Recolección completada exitosamente ===")
    logger.info(f"Archivo final: {combined_path}")

    return combined_path


if __name__ == "__main__":
    main()
