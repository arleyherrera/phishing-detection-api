"""
preprocessing.py

Módulo para preprocesamiento de datos de phishing.

Funcionalidades:
- Limpieza de datos (valores nulos, duplicados, outliers)
- Normalización y estandarización de features
- Balanceo de clases (SMOTE, undersampling, oversampling)
- Split estratificado (train/val/test)
- Validación de datos

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Imports opcionales para balanceo
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    logging.warning("Librería 'imbalanced-learn' no disponible. Balanceo limitado.")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clase para limpieza de datos.

    Maneja valores nulos, duplicados y outliers.
    """

    def __init__(self):
        """Inicializa el limpiador de datos."""
        logger.info("DataCleaner inicializado")

    def check_missing_values(self, X: pd.DataFrame) -> dict:
        """
        Verifica valores faltantes en el dataset.

        Args:
            X (pd.DataFrame): Dataset de features

        Returns:
            dict: Información sobre valores faltantes
        """
        missing_info = {
            'total_missing': int(X.isnull().sum().sum()),
            'missing_per_column': X.isnull().sum().to_dict(),
            'percentage_missing': (X.isnull().sum() / len(X) * 100).to_dict()
        }

        logger.info(f"Valores faltantes totales: {missing_info['total_missing']}")

        return missing_info

    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'mean',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Maneja valores faltantes en el dataset.

        Args:
            X (pd.DataFrame): Dataset con posibles valores faltantes
            strategy (str): Estrategia ('mean', 'median', 'most_frequent', 'constant')
            fill_value (Optional[float]): Valor para strategy='constant'

        Returns:
            pd.DataFrame: Dataset sin valores faltantes
        """
        logger.info(f"Manejando valores faltantes con estrategia: {strategy}")

        # Verificar si hay valores faltantes
        missing_count = X.isnull().sum().sum()

        if missing_count == 0:
            logger.info("No hay valores faltantes. No se requiere imputación.")
            return X.copy()

        # Imputar valores
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        logger.info(f"Valores imputados: {missing_count}")

        return X_imputed

    def remove_duplicates(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Elimina filas duplicadas.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Datos sin duplicados
        """
        logger.info("Eliminando duplicados...")

        initial_count = len(X)

        # Combinar X e y temporalmente
        df_combined = X.copy()
        df_combined['__label__'] = y.values

        # Eliminar duplicados
        df_combined = df_combined.drop_duplicates()

        # Separar nuevamente
        y_clean = df_combined['__label__']
        X_clean = df_combined.drop('__label__', axis=1)

        duplicates_removed = initial_count - len(X_clean)
        logger.info(f"Duplicados eliminados: {duplicates_removed}")

        return X_clean, y_clean

    def detect_outliers_iqr(self, X: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        """
        Detecta outliers usando método IQR (Interquartile Range).

        Args:
            X (pd.DataFrame): Dataset
            threshold (float): Multiplicador de IQR (típicamente 1.5 o 3.0)

        Returns:
            pd.DataFrame: DataFrame booleano indicando outliers
        """
        logger.info(f"Detectando outliers con IQR (threshold={threshold})...")

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        outliers = (X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))

        outlier_count = outliers.sum().sum()
        logger.info(f"Outliers detectados: {outlier_count}")

        return outliers

    def remove_outliers(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 3.0,
        method: str = 'iqr'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Elimina outliers del dataset.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            threshold (float): Umbral para detección
            method (str): Método de detección ('iqr', 'zscore')

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Datos sin outliers
        """
        logger.info(f"Eliminando outliers con método {method}...")

        initial_count = len(X)

        if method == 'iqr':
            outliers = self.detect_outliers_iqr(X, threshold)
            # Filas que NO son outliers en NINGUNA columna
            mask = ~outliers.any(axis=1)
        else:
            logger.warning(f"Método {method} no implementado. Usando IQR.")
            outliers = self.detect_outliers_iqr(X, threshold)
            mask = ~outliers.any(axis=1)

        X_clean = X[mask]
        y_clean = y[mask]

        removed_count = initial_count - len(X_clean)
        logger.info(f"Outliers eliminados: {removed_count}")
        logger.info(f"Samples restantes: {len(X_clean)}")

        return X_clean, y_clean


class FeatureScaler:
    """
    Clase para normalización y estandarización de features.

    Importante para modelos como SVM y KNN que son sensibles a escalas.
    """

    def __init__(self, scaler_type: str = 'standard'):
        """
        Inicializa el escalador de features.

        Args:
            scaler_type (str): Tipo de escalador ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Scaler tipo '{scaler_type}' no reconocido. Usando StandardScaler.")
            self.scaler = StandardScaler()

        logger.info(f"FeatureScaler inicializado con tipo: {scaler_type}")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta y transforma los datos.

        Args:
            X (pd.DataFrame): Features sin escalar

        Returns:
            pd.DataFrame: Features escaladas
        """
        logger.info(f"Escalando features con {self.scaler_type}...")

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        logger.info("Features escaladas exitosamente")

        return X_scaled_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma datos usando scaler previamente ajustado.

        Args:
            X (pd.DataFrame): Features sin escalar

        Returns:
            pd.DataFrame: Features escaladas
        """
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X_scaled_df

    def save_scaler(self, filepath: str):
        """
        Guarda el scaler en disco.

        Args:
            filepath (str): Ruta donde guardar el scaler
        """
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler guardado en: {filepath}")

    def load_scaler(self, filepath: str):
        """
        Carga un scaler desde disco.

        Args:
            filepath (str): Ruta del scaler guardado
        """
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler cargado desde: {filepath}")


class ClassBalancer:
    """
    Clase para balanceo de clases.

    Implementa técnicas de oversampling, undersampling y combinadas.
    """

    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Inicializa el balanceador de clases.

        Args:
            method (str): Método de balanceo ('smote', 'oversample', 'undersample', 'smote_tomek')
            random_state (int): Semilla aleatoria para reproducibilidad
        """
        self.method = method
        self.random_state = random_state

        if not IMBALANCED_AVAILABLE and method != 'none':
            logger.warning("imbalanced-learn no disponible. Instalando...")
            raise ImportError("pip install imbalanced-learn")

        logger.info(f"ClassBalancer inicializado con método: {method}")

    def balance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balancea las clases del dataset.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Datos balanceados
        """
        logger.info(f"Aplicando balanceo con método: {self.method}")
        logger.info(f"Distribución original: {y.value_counts().to_dict()}")

        if self.method == 'none':
            logger.info("Sin balanceo aplicado")
            return X, y

        # SMOTE (Synthetic Minority Over-sampling Technique)
        if self.method == 'smote':
            balancer = SMOTE(random_state=self.random_state)

        # Random Oversampling
        elif self.method == 'oversample':
            balancer = RandomOverSampler(random_state=self.random_state)

        # Random Undersampling
        elif self.method == 'undersample':
            balancer = RandomUnderSampler(random_state=self.random_state)

        # SMOTE + Tomek Links (combinado)
        elif self.method == 'smote_tomek':
            balancer = SMOTETomek(random_state=self.random_state)

        else:
            logger.warning(f"Método {self.method} no reconocido. Sin balanceo.")
            return X, y

        # Aplicar balanceo
        X_balanced, y_balanced = balancer.fit_resample(X, y)

        # Convertir a DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)

        logger.info(f"Distribución balanceada: {y_balanced.value_counts().to_dict()}")
        logger.info(f"Samples después de balanceo: {len(X_balanced)}")

        return X_balanced, y_balanced


class DataSplitter:
    """
    Clase para dividir datos en train/validation/test sets.

    Usa split estratificado para mantener proporción de clases.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Inicializa el divisor de datos.

        Args:
            train_size (float): Proporción para entrenamiento (default: 0.7)
            val_size (float): Proporción para validación (default: 0.15)
            test_size (float): Proporción para prueba (default: 0.15)
            random_state (int): Semilla aleatoria
        """
        # Validar que las proporciones sumen 1.0
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Las proporciones deben sumar 1.0 (actual: {total})")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        logger.info(f"DataSplitter inicializado: train={train_size}, val={val_size}, test={test_size}")

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Divide los datos en train/val/test de forma estratificada.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Dividiendo datos en train/val/test...")

        # Primer split: train+val vs test
        test_ratio = self.test_size
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=y
        )

        # Segundo split: train vs val
        val_ratio = self.val_size / (self.train_size + self.val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Verificar distribución estratificada
        logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Val distribution: {y_val.value_counts().to_dict()}")
        logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_val, X_test, y_train, y_val, y_test


class Preprocessor:
    """
    Clase principal que orquesta todo el preprocesamiento.

    Combina limpieza, normalización, balanceo y split.
    """

    def __init__(
        self,
        scaler_type: str = 'standard',
        balance_method: str = 'smote',
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
        random_state: int = 42
    ):
        """
        Inicializa el preprocesador principal.

        Args:
            scaler_type (str): Tipo de escalador
            balance_method (str): Método de balanceo
            handle_outliers (bool): ¿Eliminar outliers?
            outlier_threshold (float): Umbral para outliers
            random_state (int): Semilla aleatoria
        """
        self.cleaner = DataCleaner()
        self.scaler = FeatureScaler(scaler_type)
        self.balancer = ClassBalancer(balance_method, random_state)
        self.splitter = DataSplitter(random_state=random_state)

        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

        logger.info("Preprocessor principal inicializado")

    def preprocess_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        apply_balancing: bool = True,
        apply_scaling: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Ejecuta el pipeline completo de preprocesamiento.

        Args:
            X (pd.DataFrame): Features crudas
            y (pd.Series): Labels
            apply_balancing (bool): Aplicar balanceo de clases
            apply_scaling (bool): Aplicar escalado de features

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("=== Iniciando pipeline de preprocesamiento ===")

        # 1. Verificar y manejar valores faltantes
        missing_info = self.cleaner.check_missing_values(X)
        if missing_info['total_missing'] > 0:
            X = self.cleaner.handle_missing_values(X, strategy='mean')

        # 2. Eliminar duplicados
        X, y = self.cleaner.remove_duplicates(X, y)

        # 3. Eliminar outliers (opcional)
        if self.handle_outliers:
            X, y = self.cleaner.remove_outliers(X, y, threshold=self.outlier_threshold)

        # 4. Split en train/val/test ANTES de balancear (importante!)
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split(X, y)

        # 5. Balanceo de clases (solo en train set!)
        if apply_balancing:
            X_train, y_train = self.balancer.balance(X_train, y_train)

        # 6. Escalado de features (ajustar en train, aplicar en val y test)
        if apply_scaling:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        logger.info("=== Preprocesamiento completado ===")
        logger.info(f"Final shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_preprocessor_state(self, output_dir: str):
        """
        Guarda el estado del preprocesador (scaler, etc.).

        Args:
            output_dir (str): Directorio donde guardar
        """
        os.makedirs(output_dir, exist_ok=True)

        scaler_path = os.path.join(output_dir, "scaler.pkl")
        self.scaler.save_scaler(scaler_path)

        logger.info(f"Estado del preprocesador guardado en: {output_dir}")


def main():
    """
    Función principal para probar el preprocesador.

    Ejemplo de uso:
        python preprocessing.py
    """
    logger.info("=== Prueba de preprocesamiento ===")

    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples) * 10,
        'feature3': np.random.randint(0, 100, n_samples)
    })

    y = pd.Series(np.random.randint(0, 2, n_samples))

    print(f"Dataset original: {X.shape}")
    print(f"Distribución de clases:\n{y.value_counts()}")

    # Aplicar preprocesamiento
    preprocessor = Preprocessor(
        scaler_type='standard',
        balance_method='smote',
        handle_outliers=True
    )

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X, y,
        apply_balancing=True,
        apply_scaling=True
    )

    print(f"\n=== Después del preprocesamiento ===")
    print(f"Train: {X_train.shape}, {y_train.value_counts().to_dict()}")
    print(f"Val: {X_val.shape}, {y_val.value_counts().to_dict()}")
    print(f"Test: {X_test.shape}, {y_test.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
