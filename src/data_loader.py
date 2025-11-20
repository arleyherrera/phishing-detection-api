"""
data_loader.py

Módulo para carga y gestión de datasets de phishing.

Proporciona funcionalidades para:
- Cargar el dataset PhiUSIIL de UCI Machine Learning Repository
- Cargar datos recolectados de PhishTank y URLs peruanas
- Combinar múltiples fuentes de datos
- Validar y verificar integridad de datos

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UCIDatasetLoader:
    """
    Cargador para el dataset PhiUSIIL de UCI Machine Learning Repository.

    Dataset ID: 967
    Características: 56 features + 1 label
    URLs: ~235,000 (134,850 legítimas + 100,945 phishing)
    """

    def __init__(self, data_dir: str = "../data"):
        """
        Inicializa el cargador de dataset UCI.

        Args:
            data_dir (str): Directorio base para datos
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        logger.info(f"UCIDatasetLoader inicializado. Directorio: {data_dir}")

    def load_uci_dataset(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga el dataset PhiUSIIL de UCI.

        Args:
            force_download (bool): Forzar descarga incluso si existe localmente

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_features, y_labels)
        """
        logger.info("Cargando dataset PhiUSIIL de UCI...")

        try:
            # Importar librería UCI
            from ucimlrepo import fetch_ucirepo

            # Descargar dataset (ID: 967)
            logger.info("Descargando dataset desde UCI ML Repository...")
            phiusiil = fetch_ucirepo(id=967)

            # Extraer features y labels
            X = phiusiil.data.features
            y = phiusiil.data.targets

            # Convertir a DataFrame si no lo es
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if not isinstance(y, pd.Series):
                y = pd.Series(y.values.ravel())

            logger.info(f"Dataset UCI cargado exitosamente")
            logger.info(f"Shape de features: {X.shape}")
            logger.info(f"Shape de labels: {y.shape}")
            logger.info(f"Columnas: {list(X.columns)}")

            # Información del dataset
            if hasattr(phiusiil, 'metadata'):
                logger.info(f"Metadata: {phiusiil.metadata}")

            # Guardar localmente para uso futuro
            self._save_uci_dataset_locally(X, y)

            return X, y

        except ImportError:
            logger.error("Librería 'ucimlrepo' no instalada. Instalar con: pip install ucimlrepo")
            raise
        except Exception as e:
            logger.error(f"Error cargando dataset UCI: {e}")
            # Intentar cargar desde archivo local si existe
            return self._load_local_uci_dataset()

    def _save_uci_dataset_locally(self, X: pd.DataFrame, y: pd.Series):
        """
        Guarda el dataset UCI localmente para acceso rápido.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
        """
        try:
            features_path = os.path.join(self.raw_dir, "uci_phiusiil_features.csv")
            labels_path = os.path.join(self.raw_dir, "uci_phiusiil_labels.csv")

            X.to_csv(features_path, index=False)
            y.to_csv(labels_path, index=False, header=['label'])

            logger.info(f"Dataset UCI guardado localmente en {self.raw_dir}")
        except Exception as e:
            logger.warning(f"No se pudo guardar dataset localmente: {e}")

    def _load_local_uci_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga el dataset UCI desde archivos locales.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_features, y_labels)

        Raises:
            FileNotFoundError: Si no existen archivos locales
        """
        logger.info("Intentando cargar dataset UCI desde archivos locales...")

        features_path = os.path.join(self.raw_dir, "uci_phiusiil_features.csv")
        labels_path = os.path.join(self.raw_dir, "uci_phiusiil_labels.csv")

        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            logger.error("Archivos locales no encontrados")
            raise FileNotFoundError("Dataset UCI no encontrado localmente")

        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze()

        logger.info(f"Dataset UCI cargado desde archivos locales")
        logger.info(f"Shape: {X.shape}, {y.shape}")

        return X, y

    def get_dataset_info(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Obtiene información estadística del dataset.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            dict: Diccionario con información del dataset
        """
        info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_phishing': int(y.sum()),
            'n_legitimate': int(len(y) - y.sum()),
            'class_balance': {
                'phishing': float(y.sum() / len(y)),
                'legitimate': float((len(y) - y.sum()) / len(y))
            },
            'feature_names': list(X.columns),
            'missing_values': int(X.isnull().sum().sum()),
            'duplicates': int(X.duplicated().sum())
        }

        return info


class ExternalDataLoader:
    """
    Cargador para datos externos (PhishTank y URLs peruanas).
    """

    def __init__(self, data_dir: str = "../data/external"):
        """
        Inicializa el cargador de datos externos.

        Args:
            data_dir (str): Directorio con datos externos
        """
        self.data_dir = data_dir
        logger.info(f"ExternalDataLoader inicializado. Directorio: {data_dir}")

    def load_latest_external_data(self) -> Optional[pd.DataFrame]:
        """
        Carga el archivo más reciente de datos externos combinados.

        Returns:
            Optional[pd.DataFrame]: DataFrame con URLs y labels, o None si no existe
        """
        logger.info("Buscando datos externos...")

        if not os.path.exists(self.data_dir):
            logger.warning(f"Directorio {self.data_dir} no existe")
            return None

        # Buscar archivos features_extracted_*.csv (generados por notebook 00)
        feature_files = list(Path(self.data_dir).glob("features_extracted_*.csv"))

        if not feature_files:
            # Fallback: buscar archivos combined_urls_*.csv
            feature_files = list(Path(self.data_dir).glob("combined_urls_*.csv"))

        if not feature_files:
            logger.warning("No se encontraron archivos de datos externos")
            return None

        # Obtener el más reciente
        latest_file = max(feature_files, key=os.path.getctime)
        logger.info(f"Cargando archivo más reciente: {latest_file.name}")

        df = pd.read_csv(latest_file)
        logger.info(f"Datos externos cargados: {df.shape}")

        # Verificar que tenga columna label
        if 'label' in df.columns:
            logger.info(f"Distribución de clases:\n{df['label'].value_counts()}")
        else:
            logger.warning("Archivo no tiene columna 'label'")

        return df

    def load_phishtank_data(self) -> Optional[pd.DataFrame]:
        """
        Carga datos de PhishTank.

        Returns:
            Optional[pd.DataFrame]: DataFrame con URLs de PhishTank
        """
        logger.info("Buscando datos de PhishTank...")

        if not os.path.exists(self.data_dir):
            return None

        # Buscar archivos phishtank_*.csv
        phishtank_files = list(Path(self.data_dir).glob("phishtank_*.csv"))

        if not phishtank_files:
            logger.warning("No se encontraron archivos de PhishTank")
            return None

        latest_file = max(phishtank_files, key=os.path.getctime)
        logger.info(f"Cargando PhishTank: {latest_file}")

        df = pd.read_csv(latest_file)
        logger.info(f"URLs de PhishTank: {len(df)}")

        return df

    def load_peru_data(self) -> Optional[pd.DataFrame]:
        """
        Carga datos de URLs peruanas legítimas.

        Returns:
            Optional[pd.DataFrame]: DataFrame con URLs peruanas
        """
        logger.info("Buscando datos de URLs peruanas...")

        if not os.path.exists(self.data_dir):
            return None

        # Buscar archivos legitimate_urls_*.csv
        peru_files = list(Path(self.data_dir).glob("legitimate_urls_*.csv"))

        if not peru_files:
            logger.warning("No se encontraron archivos de URLs peruanas")
            return None

        latest_file = max(peru_files, key=os.path.getctime)
        logger.info(f"Cargando URLs peruanas: {latest_file}")

        df = pd.read_csv(latest_file)
        logger.info(f"URLs peruanas: {len(df)}")

        return df


class DatasetCombiner:
    """
    Clase para combinar datasets de UCI y datos externos.
    """

    def __init__(self):
        """Inicializa el combinador de datasets."""
        logger.info("DatasetCombiner inicializado")

    def combine_datasets(
        self,
        uci_X: pd.DataFrame,
        uci_y: pd.Series,
        external_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combina dataset UCI con datos externos (si existen).

        Args:
            uci_X (pd.DataFrame): Features de UCI
            uci_y (pd.Series): Labels de UCI
            external_df (Optional[pd.DataFrame]): Datos externos con features extraídas

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Dataset combinado (X, y)
        """
        logger.info("Combinando datasets...")

        # Si no hay datos externos, retornar UCI directamente
        if external_df is None or len(external_df) == 0:
            logger.info("No hay datos externos. Usando solo dataset UCI")
            return uci_X, uci_y

        # Verificar que external_df tenga las mismas columnas que uci_X
        if 'label' in external_df.columns:
            external_y = external_df['label']
            external_X = external_df.drop('label', axis=1)

            # Verificar columnas coincidentes
            missing_cols = set(uci_X.columns) - set(external_X.columns)
            if missing_cols:
                logger.warning(f"Columnas faltantes en datos externos: {missing_cols}")
                # Agregar columnas faltantes con valores por defecto (0)
                for col in missing_cols:
                    external_X[col] = 0

            # Asegurar mismo orden de columnas
            external_X = external_X[uci_X.columns]

            # Combinar
            X_combined = pd.concat([uci_X, external_X], ignore_index=True)
            y_combined = pd.concat([uci_y, external_y], ignore_index=True)

            logger.info(f"Datasets combinados exitosamente")
            logger.info(f"Shape final: {X_combined.shape}")
            logger.info(f"Distribución de clases:\n{y_combined.value_counts()}")

            return X_combined, y_combined
        else:
            logger.error("external_df no tiene columna 'label'")
            return uci_X, uci_y

    def balance_check(self, y: pd.Series) -> dict:
        """
        Verifica el balance de clases.

        Args:
            y (pd.Series): Labels

        Returns:
            dict: Información de balance
        """
        value_counts = y.value_counts()
        total = len(y)

        balance_info = {
            'total_samples': total,
            'class_counts': value_counts.to_dict(),
            'class_percentages': (value_counts / total * 100).to_dict(),
            'is_balanced': (value_counts.min() / value_counts.max()) > 0.8
        }

        return balance_info


class DataLoader:
    """
    Clase principal para carga de todos los datos.

    Facilita la carga y combinación de múltiples fuentes de datos.
    """

    def __init__(self, data_dir: str = "../data"):
        """
        Inicializa el cargador principal de datos.

        Args:
            data_dir (str): Directorio base de datos
        """
        self.data_dir = data_dir
        self.uci_loader = UCIDatasetLoader(data_dir)
        self.external_loader = ExternalDataLoader(os.path.join(data_dir, "external"))
        self.combiner = DatasetCombiner()
        logger.info("DataLoader principal inicializado")

    def load_all_data(
        self,
        include_external: bool = True,
        force_download_uci: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, dict]:
        """
        Carga todos los datos disponibles.

        Args:
            include_external (bool): Incluir datos externos (PhishTank + Perú)
            force_download_uci (bool): Forzar descarga de UCI

        Returns:
            Tuple[pd.DataFrame, pd.Series, dict]: (X, y, info)
        """
        logger.info("=== Iniciando carga completa de datos ===")

        # 1. Cargar dataset UCI
        uci_X, uci_y = self.uci_loader.load_uci_dataset(force_download=force_download_uci)
        uci_info = self.uci_loader.get_dataset_info(uci_X, uci_y)

        logger.info(f"Dataset UCI - Samples: {uci_info['n_samples']}, Features: {uci_info['n_features']}")

        # 2. Cargar datos externos (si se solicita)
        external_df = None
        if include_external:
            external_df = self.external_loader.load_latest_external_data()

        # 3. Combinar datasets
        X, y = self.combiner.combine_datasets(uci_X, uci_y, external_df)

        # 4. Generar información final
        balance_info = self.combiner.balance_check(y)

        info = {
            'uci_info': uci_info,
            'external_included': external_df is not None,
            'external_samples': len(external_df) if external_df is not None else 0,
            'final_shape': X.shape,
            'balance_info': balance_info
        }

        logger.info("=== Carga de datos completada ===")
        logger.info(f"Total final de samples: {len(X)}")
        logger.info(f"Balance de clases: {balance_info['class_percentages']}")

        return X, y, info


def main():
    """
    Función principal para probar el cargador de datos.

    Ejemplo de uso:
        python data_loader.py
    """
    logger.info("=== Prueba de carga de datos ===")

    # Crear cargador
    loader = DataLoader(data_dir="../data")

    # Cargar todos los datos
    X, y, info = loader.load_all_data(include_external=True)

    print("\n=== Información del Dataset ===")
    print(f"Shape: {X.shape}")
    print(f"Features: {list(X.columns)[:10]}...")  # Primeras 10
    print(f"\nDistribución de clases:")
    print(y.value_counts())
    print(f"\nBalance info:")
    print(info['balance_info'])

    # Mostrar primeras filas
    print("\n=== Primeras filas ===")
    print(X.head())


if __name__ == "__main__":
    main()
