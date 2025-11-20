"""
ensemble_models.py

Módulo para métodos de ensemble learning.

Implementa:
1. Voting Classifier (soft voting)
2. Stacking Classifier

Los métodos ensemble combinan múltiples modelos base para mejorar
el rendimiento y la robustez de las predicciones.

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
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VotingEnsemble:
    """
    Clase para Voting Classifier.

    Combina múltiples modelos base y realiza votación (soft o hard)
    para la predicción final.
    """

    def __init__(
        self,
        voting: str = 'soft',
        random_state: int = 42
    ):
        """
        Inicializa el Voting Ensemble.

        Args:
            voting (str): Tipo de votación ('soft' o 'hard')
            random_state (int): Semilla aleatoria
        """
        self.voting = voting
        self.random_state = random_state
        self.model = None
        self.base_estimators = []
        self.training_time = 0.0
        self.cv_scores = None

        logger.info(f"VotingEnsemble inicializado con voting={voting}")

    def add_estimator(self, name: str, estimator):
        """
        Agrega un modelo base al ensemble.

        Args:
            name (str): Nombre del estimador
            estimator: Modelo entrenado o por entrenar
        """
        self.base_estimators.append((name, estimator))
        logger.info(f"Estimador agregado: {name}")

    def add_estimators_from_trainers(self, trainers: Dict):
        """
        Agrega estimadores desde un diccionario de trainers.

        Args:
            trainers (Dict): Diccionario {nombre: trainer_object}
        """
        logger.info("Agregando estimadores desde trainers...")

        for name, trainer in trainers.items():
            if trainer.model is not None:
                # Convertir nombre para que sea válido
                estimator_name = name.replace(' ', '_').lower()
                self.add_estimator(estimator_name, trainer.model)
            else:
                logger.warning(f"Trainer {name} no tiene modelo entrenado. Omitiendo.")

        logger.info(f"Total de estimadores agregados: {len(self.base_estimators)}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ):
        """
        Entrena el Voting Classifier.

        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Labels de entrenamiento
            cv (int): Número de folds para cross-validation
        """
        if len(self.base_estimators) == 0:
            raise ValueError("No hay estimadores base. Usar add_estimator() primero.")

        logger.info(f"Entrenando VotingClassifier con {len(self.base_estimators)} estimadores...")
        logger.info(f"Tipo de votación: {self.voting}")

        # Crear VotingClassifier
        self.model = VotingClassifier(
            estimators=self.base_estimators,
            voting=self.voting,
            n_jobs=-1
        )

        # Entrenar
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")

        # Cross-validation
        logger.info("Ejecutando cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        self.cv_scores = cv_scores.mean()
        logger.info(f"CV F1-score: {self.cv_scores:.4f} (+/- {cv_scores.std():.4f})")

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

        return self.model.predict(X)

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

        if self.voting == 'soft':
            return self.model.predict_proba(X)
        else:
            logger.warning("predict_proba solo disponible con voting='soft'")
            return None

    def save_model(self, filepath: str):
        """
        Guarda el modelo ensemble.

        Args:
            filepath (str): Ruta donde guardar
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"VotingClassifier guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo ensemble guardado.

        Args:
            filepath (str): Ruta del modelo
        """
        self.model = joblib.load(filepath)
        logger.info(f"VotingClassifier cargado desde: {filepath}")


class StackingEnsemble:
    """
    Clase para Stacking Classifier.

    Usa predicciones de modelos base como features para un
    meta-modelo (típicamente Logistic Regression).
    """

    def __init__(
        self,
        final_estimator=None,
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Inicializa el Stacking Ensemble.

        Args:
            final_estimator: Meta-modelo (default: LogisticRegression)
            cv (int): Cross-validation folds
            random_state (int): Semilla aleatoria
        """
        self.random_state = random_state
        self.cv = cv
        self.model = None
        self.base_estimators = []
        self.training_time = 0.0
        self.cv_scores = None

        # Meta-modelo por defecto
        if final_estimator is None:
            self.final_estimator = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        else:
            self.final_estimator = final_estimator

        logger.info(f"StackingEnsemble inicializado")
        logger.info(f"Meta-modelo: {type(self.final_estimator).__name__}")

    def add_estimator(self, name: str, estimator):
        """
        Agrega un modelo base al ensemble.

        Args:
            name (str): Nombre del estimador
            estimator: Modelo entrenado o por entrenar
        """
        self.base_estimators.append((name, estimator))
        logger.info(f"Estimador agregado: {name}")

    def add_estimators_from_trainers(self, trainers: Dict):
        """
        Agrega estimadores desde un diccionario de trainers.

        Args:
            trainers (Dict): Diccionario {nombre: trainer_object}
        """
        logger.info("Agregando estimadores desde trainers...")

        for name, trainer in trainers.items():
            if trainer.model is not None:
                estimator_name = name.replace(' ', '_').lower()
                self.add_estimator(estimator_name, trainer.model)
            else:
                logger.warning(f"Trainer {name} no tiene modelo entrenado. Omitiendo.")

        logger.info(f"Total de estimadores agregados: {len(self.base_estimators)}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """
        Entrena el Stacking Classifier.

        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Labels de entrenamiento
        """
        if len(self.base_estimators) == 0:
            raise ValueError("No hay estimadores base. Usar add_estimator() primero.")

        logger.info(f"Entrenando StackingClassifier con {len(self.base_estimators)} estimadores...")
        logger.info(f"Meta-modelo: {type(self.final_estimator).__name__}")

        # Crear StackingClassifier
        self.model = StackingClassifier(
            estimators=self.base_estimators,
            final_estimator=self.final_estimator,
            cv=self.cv,
            n_jobs=-1
        )

        # Entrenar
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        logger.info(f"Tiempo de entrenamiento: {self.training_time:.2f}s")

        # Cross-validation
        logger.info("Ejecutando cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=self.cv,
            scoring='f1',
            n_jobs=-1
        )

        self.cv_scores = cv_scores.mean()
        logger.info(f"CV F1-score: {self.cv_scores:.4f} (+/- {cv_scores.std():.4f})")

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

        return self.model.predict(X)

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

        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        """
        Guarda el modelo ensemble.

        Args:
            filepath (str): Ruta donde guardar
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"StackingClassifier guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo ensemble guardado.

        Args:
            filepath (str): Ruta del modelo
        """
        self.model = joblib.load(filepath)
        logger.info(f"StackingClassifier cargado desde: {filepath}")


class EnsembleManager:
    """
    Clase para gestionar múltiples métodos ensemble.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el gestor de ensemble.

        Args:
            random_state (int): Semilla aleatoria
        """
        self.random_state = random_state
        self.voting_ensemble = None
        self.stacking_ensemble = None

        logger.info("EnsembleManager inicializado")

    def create_voting_ensemble(
        self,
        trainers: Dict,
        voting: str = 'soft'
    ) -> VotingEnsemble:
        """
        Crea un Voting Classifier desde trainers.

        Args:
            trainers (Dict): Diccionario de trainers entrenados
            voting (str): Tipo de votación

        Returns:
            VotingEnsemble: Ensemble configurado
        """
        logger.info("Creando VotingEnsemble...")

        self.voting_ensemble = VotingEnsemble(
            voting=voting,
            random_state=self.random_state
        )

        self.voting_ensemble.add_estimators_from_trainers(trainers)

        return self.voting_ensemble

    def create_stacking_ensemble(
        self,
        trainers: Dict,
        final_estimator=None
    ) -> StackingEnsemble:
        """
        Crea un Stacking Classifier desde trainers.

        Args:
            trainers (Dict): Diccionario de trainers entrenados
            final_estimator: Meta-modelo

        Returns:
            StackingEnsemble: Ensemble configurado
        """
        logger.info("Creando StackingEnsemble...")

        self.stacking_ensemble = StackingEnsemble(
            final_estimator=final_estimator,
            random_state=self.random_state
        )

        self.stacking_ensemble.add_estimators_from_trainers(trainers)

        return self.stacking_ensemble

    def train_all_ensembles(
        self,
        trainers: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ) -> Dict[str, object]:
        """
        Crea y entrena todos los ensembles.

        Args:
            trainers (Dict): Trainers de modelos base
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Labels de entrenamiento
            cv (int): Cross-validation folds

        Returns:
            Dict[str, object]: Diccionario de ensembles entrenados
        """
        logger.info("=== Entrenando todos los ensembles ===")

        ensembles = {}

        # 1. Voting Classifier
        logger.info("\n" + "="*60)
        logger.info("VOTING CLASSIFIER (Soft Voting)")
        logger.info("="*60)

        voting = self.create_voting_ensemble(trainers, voting='soft')
        voting.train(X_train, y_train, cv=cv)
        ensembles['voting_soft'] = voting

        # 2. Stacking Classifier
        logger.info("\n" + "="*60)
        logger.info("STACKING CLASSIFIER")
        logger.info("="*60)

        stacking = self.create_stacking_ensemble(trainers)
        stacking.train(X_train, y_train)
        ensembles['stacking'] = stacking

        logger.info("\n=== Entrenamiento de ensembles completado ===")

        return ensembles

    def save_ensembles(
        self,
        ensembles: Dict,
        output_dir: str
    ):
        """
        Guarda todos los ensembles.

        Args:
            ensembles (Dict): Diccionario de ensembles
            output_dir (str): Directorio de salida
        """
        logger.info(f"Guardando ensembles en: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        for name, ensemble in ensembles.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            ensemble.save_model(filepath)

        logger.info("Todos los ensembles guardados exitosamente")

    def compare_ensembles_vs_base(
        self,
        base_trainers: Dict,
        ensembles: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compara el rendimiento de ensembles vs modelos base.

        Args:
            base_trainers (Dict): Trainers de modelos base
            ensembles (Dict): Ensembles entrenados
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Labels de test

        Returns:
            pd.DataFrame: Tabla comparativa
        """
        logger.info("Comparando ensembles vs modelos base...")

        results = []

        # Evaluar modelos base
        for name, trainer in base_trainers.items():
            y_pred = trainer.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Type': 'Base',
                'Accuracy': accuracy,
                'F1-Score': f1
            })

        # Evaluar ensembles
        for name, ensemble in ensembles.items():
            y_pred = ensemble.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Type': 'Ensemble',
                'Accuracy': accuracy,
                'F1-Score': f1
            })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('F1-Score', ascending=False)

        logger.info("\n=== Comparación Ensembles vs Base Models ===")
        logger.info(f"\n{df_results.to_string(index=False)}")

        return df_results


def main():
    """
    Función principal para probar ensemble methods.

    Ejemplo de uso:
        python ensemble_models.py
    """
    logger.info("=== Prueba de Ensemble Methods ===")

    # Datos de ejemplo
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y)

    # Crear modelos base simples
    class SimpleTrainer:
        def __init__(self, model):
            self.model = model
            self.model_name = type(model).__name__

    # Entrenar modelos base
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42, n_estimators=50)
    lr = LogisticRegression(random_state=42, max_iter=1000)

    dt.fit(X_df, y_series)
    rf.fit(X_df, y_series)
    lr.fit(X_df, y_series)

    trainers = {
        'decision_tree': SimpleTrainer(dt),
        'random_forest': SimpleTrainer(rf),
        'logistic_regression': SimpleTrainer(lr)
    }

    # Crear y entrenar ensembles
    manager = EnsembleManager(random_state=42)

    ensembles = manager.train_all_ensembles(
        trainers,
        X_df,
        y_series,
        cv=5
    )

    print(f"\nEnsembles creados: {list(ensembles.keys())}")
    print(f"Voting CV Score: {ensembles['voting_soft'].cv_scores:.4f}")
    print(f"Stacking CV Score: {ensembles['stacking'].cv_scores:.4f}")


if __name__ == "__main__":
    main()
