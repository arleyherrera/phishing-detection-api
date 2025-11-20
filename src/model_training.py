"""
model_training.py

Módulo para entrenamiento de modelos de Machine Learning.

Implementa 8 modelos de clasificación:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Gradient Boosting
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors (KNN)
8. Naive Bayes

Con:
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Cross-validation
- Guardado de modelos entrenados
- Registro de métricas

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import pandas as pd
import numpy as np
import logging
import time
import joblib
import os
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost no disponible. Instalar con: pip install xgboost")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """
    Clase base para entrenadores de modelos.

    Proporciona funcionalidades comunes para todos los modelos.
    """

    def __init__(self, model_name: str, random_state: int = 42):
        """
        Inicializa el entrenador base.

        Args:
            model_name (str): Nombre del modelo
            random_state (int): Semilla aleatoria
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.training_time = 0.0
        self.inference_time = 0.0

        logger.info(f"{model_name} Trainer inicializado")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Entrena el modelo (debe ser implementado por subclases).

        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Labels de entrenamiento
        """
        raise NotImplementedError("Subclases deben implementar train()")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

        start_time = time.time()
        predictions = self.model.predict(X)
        self.inference_time = time.time() - start_time

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades de predicción.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"{self.model_name} no soporta predict_proba")
            return None

    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado.

        Args:
            filepath (str): Ruta donde guardar
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Modelo {self.model_name} guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo guardado.

        Args:
            filepath (str): Ruta del modelo
        """
        self.model = joblib.load(filepath)
        logger.info(f"Modelo {self.model_name} cargado desde: {filepath}")


class LogisticRegressionTrainer(BaseModelTrainer):
    """Entrenador para Logistic Regression."""

    def __init__(self, random_state: int = 42):
        super().__init__("Logistic Regression", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena Logistic Regression con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'C': [1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear'],
                'max_iter': [500]
            }

            base_model = LogisticRegression(random_state=self.random_state)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_scores = grid_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
            logger.info(f"Mejor CV score: {self.cv_scores:.4f}")
        else:
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class DecisionTreeTrainer(BaseModelTrainer):
    """Entrenador para Decision Tree."""

    def __init__(self, random_state: int = 42):
        super().__init__("Decision Tree", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena Decision Tree con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }

            base_model = DecisionTreeClassifier(random_state=self.random_state)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_scores = grid_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = DecisionTreeClassifier(random_state=self.random_state)
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class RandomForestTrainer(BaseModelTrainer):
    """Entrenador para Random Forest."""

    def __init__(self, random_state: int = 42):
        super().__init__("Random Forest", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena Random Forest con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10, 20],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': ['sqrt']
            }

            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

            # Usar RandomizedSearchCV para ser más rápido
            random_search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            start_time = time.time()
            random_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            self.cv_scores = random_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class XGBoostTrainer(BaseModelTrainer):
    """Entrenador para XGBoost."""

    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena XGBoost con hyperparameter tuning."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no instalado")

        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'max_depth': [3],
                'learning_rate': [0.1],
                'n_estimators': [100],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }

            base_model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            random_search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            start_time = time.time()
            random_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            self.cv_scores = random_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class GradientBoostingTrainer(BaseModelTrainer):
    """Entrenador para Gradient Boosting."""

    def __init__(self, random_state: int = 42):
        super().__init__("Gradient Boosting", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena Gradient Boosting con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }

            base_model = GradientBoostingClassifier(random_state=self.random_state)

            random_search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=15,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            start_time = time.time()
            random_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            self.cv_scores = random_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = GradientBoostingClassifier(random_state=self.random_state)
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class SVMTrainer(BaseModelTrainer):
    """Entrenador para SVM."""

    def __init__(self, random_state: int = 42):
        super().__init__("SVM", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena SVM con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'C': [1],
                'kernel': ['linear'],
                'gamma': ['scale']
            }

            base_model = SVC(random_state=self.random_state, probability=True)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_scores = grid_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = SVC(random_state=self.random_state, probability=True)
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class KNNTrainer(BaseModelTrainer):
    """Entrenador para K-Nearest Neighbors."""

    def __init__(self, random_state: int = 42):
        super().__init__("KNN", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        cv: int = 5
    ):
        """Entrena KNN con hyperparameter tuning."""
        logger.info(f"Entrenando {self.model_name}...")

        if use_grid_search:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            base_model = KNeighborsClassifier(n_jobs=-1)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            self.training_time = time.time() - start_time

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_scores = grid_search.best_score_

            logger.info(f"Mejores parámetros: {self.best_params}")
        else:
            self.model = KNeighborsClassifier()
            start_time = time.time()
            self.model.fit(X_train, y_train)
            self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class NaiveBayesTrainer(BaseModelTrainer):
    """Entrenador para Naive Bayes."""

    def __init__(self, random_state: int = 42):
        super().__init__("Naive Bayes", random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = False,
        cv: int = 5
    ):
        """Entrena Naive Bayes (sin hyperparameters significativos)."""
        logger.info(f"Entrenando {self.model_name}...")

        self.model = GaussianNB()

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        # Cross-validation manual
        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
        self.cv_scores = scores.mean()

        logger.info(f"CV F1-score: {self.cv_scores:.4f}")
        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")


class ModelTrainingManager:
    """
    Clase principal para gestionar el entrenamiento de todos los modelos.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el gestor de entrenamiento.

        Args:
            random_state (int): Semilla aleatoria
        """
        self.random_state = random_state
        self.trainers = {
            'logistic_regression': LogisticRegressionTrainer(random_state),
            'decision_tree': DecisionTreeTrainer(random_state),
            'random_forest': RandomForestTrainer(random_state),
            'xgboost': XGBoostTrainer(random_state) if XGBOOST_AVAILABLE else None,
            'gradient_boosting': GradientBoostingTrainer(random_state),
            'svm': SVMTrainer(random_state),
            'knn': KNNTrainer(random_state),
            'naive_bayes': NaiveBayesTrainer(random_state)
        }

        # Eliminar None si XGBoost no está disponible
        self.trainers = {k: v for k, v in self.trainers.items() if v is not None}

        logger.info(f"ModelTrainingManager inicializado con {len(self.trainers)} modelos")

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_grid_search: bool = True,
        models_to_train: Optional[list] = None
    ) -> Dict[str, BaseModelTrainer]:
        """
        Entrena todos los modelos.

        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Labels de entrenamiento
            use_grid_search (bool): Usar hyperparameter tuning
            models_to_train (Optional[list]): Lista de modelos específicos a entrenar

        Returns:
            Dict[str, BaseModelTrainer]: Diccionario de entrenadores
        """
        logger.info("=== Iniciando entrenamiento de todos los modelos ===")

        # Determinar qué modelos entrenar
        if models_to_train:
            trainers_to_use = {k: v for k, v in self.trainers.items() if k in models_to_train}
        else:
            trainers_to_use = self.trainers

        # Entrenar cada modelo
        for name, trainer in trainers_to_use.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Entrenando: {name.upper().replace('_', ' ')}")
            logger.info(f"{'='*60}")

            try:
                trainer.train(X_train, y_train, use_grid_search=use_grid_search)
                logger.info(f"✓ {name} entrenado exitosamente")
            except Exception as e:
                logger.error(f"✗ Error entrenando {name}: {e}")

        logger.info("\n=== Entrenamiento de modelos completado ===")

        return trainers_to_use

    def save_all_models(self, trainers: Dict, output_dir: str):
        """
        Guarda todos los modelos entrenados.

        Args:
            trainers (Dict): Diccionario de entrenadores
            output_dir (str): Directorio de salida
        """
        logger.info(f"Guardando modelos en: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        for name, trainer in trainers.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            trainer.save_model(filepath)

        logger.info("Todos los modelos guardados exitosamente")


def main():
    """
    Función principal para probar el entrenamiento.

    Ejemplo de uso:
        python model_training.py
    """
    logger.info("=== Prueba de entrenamiento de modelos ===")

    # Datos de ejemplo
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_train = pd.Series(y)

    print(f"Dataset: {X_train.shape}")

    # Entrenar modelos
    manager = ModelTrainingManager(random_state=42)

    # Entrenar solo algunos modelos para la prueba
    trainers = manager.train_all_models(
        X_train,
        y_train,
        use_grid_search=False,  # Sin grid search para ser rápido
        models_to_train=['logistic_regression', 'decision_tree', 'naive_bayes']
    )

    print(f"\nModelos entrenados: {list(trainers.keys())}")


if __name__ == "__main__":
    main()
