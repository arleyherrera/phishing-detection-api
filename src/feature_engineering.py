"""
feature_engineering.py

Módulo para ingeniería de características (feature engineering).

Funcionalidades:
- Análisis de importancia de features
- Selección de features relevantes
- Eliminación de features redundantes o con baja varianza
- Análisis de correlación
- Creación de nuevas features (si aplica)

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    VarianceThreshold,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Clase para analizar la importancia de features.

    Usa múltiples métodos para evaluar qué features son más relevantes
    para la predicción.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el analizador de importancia.

        Args:
            random_state (int): Semilla aleatoria
        """
        self.random_state = random_state
        self.importance_scores = {}
        logger.info("FeatureImportanceAnalyzer inicializado")

    def calculate_tree_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100
    ) -> pd.Series:
        """
        Calcula importancia usando Random Forest.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            n_estimators (int): Número de árboles

        Returns:
            pd.Series: Importancia de cada feature
        """
        logger.info("Calculando importancia con Random Forest...")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X, y)

        importance = pd.Series(
            rf.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        self.importance_scores['random_forest'] = importance

        logger.info(f"Top 5 features más importantes:\n{importance.head()}")

        return importance

    def calculate_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.Series:
        """
        Calcula importancia usando Mutual Information.

        La información mutua mide la dependencia entre features y target.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            pd.Series: Scores de mutual information
        """
        logger.info("Calculando Mutual Information...")

        mi_scores = mutual_info_classif(
            X, y,
            random_state=self.random_state
        )

        importance = pd.Series(
            mi_scores,
            index=X.columns
        ).sort_values(ascending=False)

        self.importance_scores['mutual_info'] = importance

        return importance

    def calculate_anova_f_score(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.Series:
        """
        Calcula importancia usando ANOVA F-statistic.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels

        Returns:
            pd.Series: F-scores de cada feature
        """
        logger.info("Calculando ANOVA F-scores...")

        f_scores, _ = f_classif(X, y)

        importance = pd.Series(
            f_scores,
            index=X.columns
        ).sort_values(ascending=False)

        self.importance_scores['anova_f'] = importance

        return importance

    def get_top_features(
        self,
        method: str = 'random_forest',
        top_k: int = 20
    ) -> List[str]:
        """
        Obtiene las top K features más importantes.

        Args:
            method (str): Método de importancia
            top_k (int): Número de features a retornar

        Returns:
            List[str]: Lista de nombres de features
        """
        if method not in self.importance_scores:
            logger.error(f"Método {method} no calculado. Ejecutar calculate_* primero.")
            return []

        top_features = self.importance_scores[method].head(top_k).index.tolist()

        logger.info(f"Top {top_k} features ({method}): {top_features[:5]}...")

        return top_features

    def visualize_importance(
        self,
        method: str = 'random_forest',
        top_k: int = 20,
        output_path: Optional[str] = None
    ):
        """
        Visualiza la importancia de features.

        Args:
            method (str): Método de importancia a visualizar
            top_k (int): Número de features a mostrar
            output_path (Optional[str]): Ruta para guardar la figura
        """
        if method not in self.importance_scores:
            logger.error(f"Método {method} no calculado.")
            return

        importance = self.importance_scores[method].head(top_k)

        plt.figure(figsize=(12, 8))
        importance.plot(kind='barh', color='steelblue')
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_k} Features - {method.replace("_", " ").title()}', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfica guardada en: {output_path}")

        plt.close()


class FeatureSelector:
    """
    Clase para selección de features.

    Implementa varios métodos de selección para reducir dimensionalidad
    y mejorar el rendimiento del modelo.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el selector de features.

        Args:
            random_state (int): Semilla aleatoria
        """
        self.random_state = random_state
        self.selected_features = []
        logger.info("FeatureSelector inicializado")

    def remove_low_variance_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Elimina features con baja varianza.

        Features con poca varianza no aportan información útil.

        Args:
            X (pd.DataFrame): Features
            threshold (float): Umbral de varianza mínima

        Returns:
            pd.DataFrame: Dataset sin features de baja varianza
        """
        logger.info(f"Eliminando features con varianza < {threshold}...")

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        # Obtener features seleccionadas
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        removed_count = len(X.columns) - len(selected_features)
        logger.info(f"Features eliminadas por baja varianza: {removed_count}")

        return X[selected_features]

    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Elimina features altamente correlacionadas.

        Cuando dos features están muy correlacionadas, una es redundante.

        Args:
            X (pd.DataFrame): Features
            threshold (float): Umbral de correlación (0.95 = 95%)

        Returns:
            pd.DataFrame: Dataset sin features redundantes
        """
        logger.info(f"Eliminando features correlacionadas (>{threshold})...")

        # Calcular matriz de correlación
        corr_matrix = X.corr().abs()

        # Crear máscara triangular superior
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

        # Encontrar features altamente correlacionadas
        to_drop = [
            column for column in corr_matrix.columns
            if any(corr_matrix[column][upper_triangle[:, corr_matrix.columns.get_loc(column)]] > threshold)
        ]

        logger.info(f"Features eliminadas por alta correlación: {len(to_drop)}")
        if to_drop:
            logger.info(f"Features eliminadas: {to_drop[:5]}...")

        return X.drop(columns=to_drop)

    def select_k_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 30,
        score_func=f_classif
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Selecciona las K mejores features usando score function.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            k (int): Número de features a seleccionar
            score_func: Función de scoring (f_classif, mutual_info_classif)

        Returns:
            Tuple[pd.DataFrame, List[str]]: (Dataset seleccionado, nombres de features)
        """
        logger.info(f"Seleccionando top {k} features con SelectKBest...")

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        # Obtener features seleccionadas
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        logger.info(f"Features seleccionadas: {len(selected_features)}")

        self.selected_features = selected_features

        return X[selected_features], selected_features

    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: int = 30
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Selección recursiva de features (RFE).

        Elimina iterativamente las features menos importantes.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            n_features_to_select (int): Número de features a seleccionar

        Returns:
            Tuple[pd.DataFrame, List[str]]: (Dataset seleccionado, nombres)
        """
        logger.info(f"Aplicando RFE para seleccionar {n_features_to_select} features...")

        # Usar Logistic Regression como estimador
        estimator = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=1
        )

        selector.fit(X, y)

        # Features seleccionadas
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        logger.info(f"RFE completado. Features seleccionadas: {len(selected_features)}")

        self.selected_features = selected_features

        return X[selected_features], selected_features


class CorrelationAnalyzer:
    """
    Clase para analizar correlaciones entre features.
    """

    def __init__(self):
        """Inicializa el analizador de correlación."""
        logger.info("CorrelationAnalyzer inicializado")

    def calculate_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula matriz de correlación.

        Args:
            X (pd.DataFrame): Features

        Returns:
            pd.DataFrame: Matriz de correlación
        """
        logger.info("Calculando matriz de correlación...")
        return X.corr()

    def visualize_correlation_heatmap(
        self,
        X: pd.DataFrame,
        top_k: Optional[int] = None,
        output_path: Optional[str] = None
    ):
        """
        Visualiza heatmap de correlación.

        Args:
            X (pd.DataFrame): Features
            top_k (Optional[int]): Mostrar solo top K features
            output_path (Optional[str]): Ruta para guardar
        """
        logger.info("Generando heatmap de correlación...")

        # Limitar a top K si se especifica
        if top_k and top_k < len(X.columns):
            # Usar las primeras K columnas
            X_subset = X.iloc[:, :top_k]
        else:
            X_subset = X

        corr_matrix = X_subset.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Matriz de Correlación de Features', fontsize=16)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap guardado en: {output_path}")

        plt.close()

    def find_highly_correlated_pairs(
        self,
        X: pd.DataFrame,
        threshold: float = 0.9
    ) -> List[Tuple[str, str, float]]:
        """
        Encuentra pares de features altamente correlacionadas.

        Args:
            X (pd.DataFrame): Features
            threshold (float): Umbral de correlación

        Returns:
            List[Tuple[str, str, float]]: Lista de (feature1, feature2, correlación)
        """
        logger.info(f"Buscando pares con correlación > {threshold}...")

        corr_matrix = X.corr().abs()

        # Obtener pares
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        logger.info(f"Pares altamente correlacionados encontrados: {len(pairs)}")

        return pairs


class FeatureEngineer:
    """
    Clase principal para orquestar toda la ingeniería de features.
    """

    def __init__(self, random_state: int = 42):
        """
        Inicializa el ingeniero de features.

        Args:
            random_state (int): Semilla aleatoria
        """
        self.importance_analyzer = FeatureImportanceAnalyzer(random_state)
        self.feature_selector = FeatureSelector(random_state)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.random_state = random_state

        logger.info("FeatureEngineer principal inicializado")

    def analyze_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Análisis completo de features.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            output_dir (Optional[str]): Directorio para guardar visualizaciones

        Returns:
            Dict: Resultados del análisis
        """
        logger.info("=== Iniciando análisis completo de features ===")

        results = {}

        # 1. Calcular importancia con diferentes métodos
        rf_importance = self.importance_analyzer.calculate_tree_importance(X, y)
        mi_importance = self.importance_analyzer.calculate_mutual_information(X, y)
        anova_importance = self.importance_analyzer.calculate_anova_f_score(X, y)

        results['importance_rf'] = rf_importance
        results['importance_mi'] = mi_importance
        results['importance_anova'] = anova_importance

        # 2. Análisis de correlación
        corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(X)
        highly_correlated = self.correlation_analyzer.find_highly_correlated_pairs(X, threshold=0.9)

        results['correlation_matrix'] = corr_matrix
        results['highly_correlated_pairs'] = highly_correlated

        # 3. Visualizaciones (si se especifica output_dir)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Importancia RF
            self.importance_analyzer.visualize_importance(
                method='random_forest',
                top_k=20,
                output_path=os.path.join(output_dir, 'feature_importance_rf.png')
            )

            # Importancia MI
            self.importance_analyzer.visualize_importance(
                method='mutual_info',
                top_k=20,
                output_path=os.path.join(output_dir, 'feature_importance_mi.png')
            )

            # Heatmap de correlación
            self.correlation_analyzer.visualize_correlation_heatmap(
                X,
                top_k=30,
                output_path=os.path.join(output_dir, 'correlation_heatmap.png')
            )

        logger.info("=== Análisis de features completado ===")

        return results

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'tree_importance',
        k: int = 40
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Selecciona las mejores features.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            method (str): Método de selección
            k (int): Número de features a seleccionar

        Returns:
            Tuple[pd.DataFrame, List[str]]: (X seleccionado, nombres)
        """
        logger.info(f"Seleccionando {k} features con método: {method}")

        if method == 'tree_importance':
            # Usar importancia de Random Forest
            importance = self.importance_analyzer.calculate_tree_importance(X, y)
            top_features = importance.head(k).index.tolist()
            return X[top_features], top_features

        elif method == 'mutual_info':
            # Usar mutual information
            return self.feature_selector.select_k_best_features(
                X, y, k=k, score_func=mutual_info_classif
            )

        elif method == 'anova':
            # Usar ANOVA F-score
            return self.feature_selector.select_k_best_features(
                X, y, k=k, score_func=f_classif
            )

        elif method == 'rfe':
            # Recursive Feature Elimination
            return self.feature_selector.recursive_feature_elimination(X, y, k)

        else:
            logger.warning(f"Método {method} no reconocido. Usando tree_importance.")
            importance = self.importance_analyzer.calculate_tree_importance(X, y)
            top_features = importance.head(k).index.tolist()
            return X[top_features], top_features


def main():
    """
    Función principal para probar feature engineering.

    Ejemplo de uso:
        python feature_engineering.py
    """
    logger.info("=== Prueba de Feature Engineering ===")

    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    y = pd.Series(np.random.randint(0, 2, n_samples))

    print(f"Dataset: {X.shape}")

    # Ingeniería de features
    engineer = FeatureEngineer(random_state=42)

    # Análisis
    results = engineer.analyze_features(X, y, output_dir='../results/visualizations')

    print(f"\nTop 10 features (Random Forest):")
    print(results['importance_rf'].head(10))

    # Selección
    X_selected, selected_features = engineer.select_features(
        X, y,
        method='tree_importance',
        k=20
    )

    print(f"\nFeatures seleccionadas: {len(selected_features)}")
    print(f"Shape final: {X_selected.shape}")


if __name__ == "__main__":
    main()
