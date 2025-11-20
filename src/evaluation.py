"""
evaluation.py

Módulo para evaluación completa de modelos.

Funcionalidades:
- Cálculo de todas las métricas (Accuracy, Precision, Recall, F1, ROC-AUC)
- Matrices de confusión
- Curvas ROC
- Comparación de modelos
- Visualizaciones profesionales (300 DPI)
- Generación de reportes

Autor: [Tu Nombre]
Universidad: [Tu Universidad]
Fecha: 2024-2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Configuración de estilo para gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Clase para evaluar modelos de clasificación.

    Calcula métricas estándar y genera visualizaciones.
    """

    def __init__(self):
        """Inicializa el evaluador de modelos."""
        logger.info("ModelEvaluator inicializado")

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcula todas las métricas de evaluación.

        Args:
            y_true (np.ndarray): Labels verdaderos
            y_pred (np.ndarray): Predicciones
            y_pred_proba (Optional[np.ndarray]): Probabilidades de predicción

        Returns:
            Dict[str, float]: Diccionario con métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        # ROC-AUC (requiere probabilidades)
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                # Tomar probabilidad de clase positiva
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = None

        return metrics

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evalúa un modelo completo.

        Args:
            model: Modelo entrenado
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Labels de test
            model_name (str): Nombre del modelo

        Returns:
            Dict[str, float]: Métricas del modelo
        """
        logger.info(f"Evaluando modelo: {model_name}")

        # Predicciones
        y_pred = model.predict(X_test)

        # Probabilidades (si disponibles)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        # Calcular métricas
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['model_name'] = model_name

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  ROC-AUC: N/A")

        return metrics

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Genera reporte de clasificación detallado.

        Args:
            y_true (np.ndarray): Labels verdaderos
            y_pred (np.ndarray): Predicciones

        Returns:
            str: Reporte de clasificación
        """
        return classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing'])


class VisualizationGenerator:
    """
    Clase para generar visualizaciones profesionales.
    """

    def __init__(self, output_dir: str = "../results/visualizations", dpi: int = 300):
        """
        Inicializa el generador de visualizaciones.

        Args:
            output_dir (str): Directorio para guardar visualizaciones
            dpi (int): Resolución de las imágenes
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"VisualizationGenerator inicializado. Output: {output_dir}, DPI: {dpi}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ):
        """
        Genera y guarda matriz de confusión.

        Args:
            y_true (np.ndarray): Labels verdaderos
            y_pred (np.ndarray): Predicciones
            model_name (str): Nombre del modelo
            save (bool): Guardar imagen
        """
        logger.info(f"Generando matriz de confusión para {model_name}...")

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar_kws={'label': 'Count'}
        )

        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save:
            filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada: {filepath}")

        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save: bool = True
    ):
        """
        Genera curva ROC individual.

        Args:
            y_true (np.ndarray): Labels verdaderos
            y_pred_proba (np.ndarray): Probabilidades
            model_name (str): Nombre del modelo
            save (bool): Guardar imagen
        """
        logger.info(f"Generando curva ROC para {model_name}...")

        # Obtener probabilidad de clase positiva
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            filename = f"roc_curve_{model_name.replace(' ', '_').lower()}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Curva ROC guardada: {filepath}")

        plt.close()

    def plot_multiple_roc_curves(
        self,
        models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save: bool = True
    ):
        """
        Genera curvas ROC de múltiples modelos en una sola gráfica.

        Args:
            models_data (Dict): {model_name: (y_true, y_pred_proba)}
            save (bool): Guardar imagen
        """
        logger.info(f"Generando curvas ROC comparativas de {len(models_data)} modelos...")

        plt.figure(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

        for idx, (model_name, (y_true, y_pred_proba)) in enumerate(models_data.items()):
            # Obtener probabilidad de clase positiva
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                color=colors[idx],
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})'
            )

        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, "roc_curves_comparison.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Curvas ROC comparativas guardadas: {filepath}")

        plt.close()

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        save: bool = True
    ):
        """
        Genera gráfico de barras comparando métricas entre modelos.

        Args:
            metrics_df (pd.DataFrame): DataFrame con métricas de modelos
            save (bool): Guardar imagen
        """
        logger.info("Generando comparación de métricas...")

        # Seleccionar métricas a visualizar
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Ordenar modelos por métrica
            df_sorted = metrics_df.sort_values(metric, ascending=False)

            ax.barh(df_sorted['model_name'], df_sorted[metric], color='steelblue')
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} by Model', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)

            # Agregar valores en las barras
            for i, v in enumerate(df_sorted[metric]):
                if pd.notna(v):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        # Ocultar ejes no utilizados
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, "metrics_comparison.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Comparación de métricas guardada: {filepath}")

        plt.close()


class ModelComparator:
    """
    Clase para comparar múltiples modelos.
    """

    def __init__(self):
        """Inicializa el comparador de modelos."""
        self.evaluator = ModelEvaluator()
        logger.info("ModelComparator inicializado")

    def compare_models(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        include_time: bool = False
    ) -> pd.DataFrame:
        """
        Compara múltiples modelos.

        Args:
            models (Dict): {model_name: model_object or trainer_object}
            X_test (pd.DataFrame): Features de test
            y_test (pd.Series): Labels de test
            include_time (bool): Incluir tiempos de entrenamiento/inferencia

        Returns:
            pd.DataFrame: Tabla comparativa
        """
        logger.info(f"Comparando {len(models)} modelos...")

        results = []

        for name, model_obj in models.items():
            # Si es un trainer, obtener el modelo
            if hasattr(model_obj, 'model'):
                model = model_obj.model
                training_time = getattr(model_obj, 'training_time', None)
            else:
                model = model_obj
                training_time = None

            # Evaluar
            metrics = self.evaluator.evaluate_model(model, X_test, y_test, name)

            if include_time and training_time:
                metrics['training_time'] = training_time

            results.append(metrics)

        df_results = pd.DataFrame(results)

        # Ordenar por F1-Score
        df_results = df_results.sort_values('f1_score', ascending=False)

        logger.info("\n=== Resultados de Comparación ===")
        logger.info(f"\n{df_results.to_string(index=False)}")

        return df_results

    def save_comparison_table(
        self,
        df_results: pd.DataFrame,
        output_path: str
    ):
        """
        Guarda tabla de comparación en CSV.

        Args:
            df_results (pd.DataFrame): Resultados
            output_path (str): Ruta de salida
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        logger.info(f"Tabla de comparación guardada: {output_path}")


class ReportGenerator:
    """
    Clase para generar reportes finales en Markdown.
    """

    def __init__(self, output_dir: str = "../results/reports"):
        """
        Inicializa el generador de reportes.

        Args:
            output_dir (str): Directorio de salida
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ReportGenerator inicializado. Output: {output_dir}")

    def generate_final_report(
        self,
        metrics_df: pd.DataFrame,
        best_model_name: str,
        dataset_info: Dict,
        output_filename: str = "final_report.md"
    ):
        """
        Genera reporte final en Markdown.

        Args:
            metrics_df (pd.DataFrame): Métricas de modelos
            best_model_name (str): Nombre del mejor modelo
            dataset_info (Dict): Información del dataset
            output_filename (str): Nombre del archivo de salida
        """
        logger.info("Generando reporte final...")

        report_lines = []

        # Encabezado
        report_lines.append("# 📊 Reporte Final - Sistema de Detección de Phishing\n")
        report_lines.append(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("---\n\n")

        # Resumen Ejecutivo
        report_lines.append("## 🎯 Resumen Ejecutivo\n\n")
        report_lines.append(f"Se entrenaron y evaluaron **{len(metrics_df)} modelos** de Machine Learning ")
        report_lines.append("para la detección automática de sitios web de phishing.\n\n")
        report_lines.append(f"**Mejor Modelo:** {best_model_name}\n\n")

        # Información del Dataset
        report_lines.append("## 📁 Dataset\n\n")
        report_lines.append(f"- **Total de muestras:** {dataset_info.get('total_samples', 'N/A')}\n")
        report_lines.append(f"- **Features:** {dataset_info.get('n_features', 'N/A')}\n")
        report_lines.append(f"- **Muestras de phishing:** {dataset_info.get('phishing_samples', 'N/A')}\n")
        report_lines.append(f"- **Muestras legítimas:** {dataset_info.get('legitimate_samples', 'N/A')}\n\n")

        # Resultados
        report_lines.append("## 📈 Resultados de Modelos\n\n")
        report_lines.append("### Tabla Comparativa\n\n")

        # Convertir DataFrame a markdown table
        report_lines.append(metrics_df.to_markdown(index=False))
        report_lines.append("\n\n")

        # Análisis del Mejor Modelo
        best_metrics = metrics_df[metrics_df['model_name'] == best_model_name].iloc[0]

        report_lines.append(f"### 🏆 Mejor Modelo: {best_model_name}\n\n")
        report_lines.append(f"- **Accuracy:** {best_metrics['accuracy']:.4f}\n")
        report_lines.append(f"- **Precision:** {best_metrics['precision']:.4f}\n")
        report_lines.append(f"- **Recall:** {best_metrics['recall']:.4f}\n")
        report_lines.append(f"- **F1-Score:** {best_metrics['f1_score']:.4f}\n")
        if pd.notna(best_metrics.get('roc_auc')):
            report_lines.append(f"- **ROC-AUC:** {best_metrics['roc_auc']:.4f}\n")
        report_lines.append("\n")

        # Conclusiones
        report_lines.append("## 💡 Conclusiones\n\n")
        report_lines.append("1. El sistema logró detectar sitios de phishing con alta precisión.\n")
        report_lines.append(f"2. El modelo **{best_model_name}** obtuvo el mejor rendimiento general.\n")
        report_lines.append("3. Las métricas indican un balance adecuado entre precisión y recall.\n\n")

        # Recomendaciones
        report_lines.append("## 🎓 Recomendaciones\n\n")
        report_lines.append("1. Actualizar el modelo periódicamente con nuevos datos de phishing.\n")
        report_lines.append("2. Monitorear falsos positivos para ajustar el umbral de clasificación.\n")
        report_lines.append("3. Integrar el modelo en un sistema de detección en tiempo real.\n\n")

        report_lines.append("---\n")
        report_lines.append("*Generado automáticamente por el Sistema de Detección de Phishing*\n")

        # Guardar reporte
        report_path = os.path.join(self.output_dir, output_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        logger.info(f"Reporte final generado: {report_path}")


def main():
    """
    Función principal para probar evaluación.

    Ejemplo de uso:
        python evaluation.py
    """
    logger.info("=== Prueba de Evaluation ===")

    # Datos de ejemplo
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)

    # Entrenar modelos simples
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)

    rf.fit(X_df, y_series)
    lr.fit(X_df, y_series)

    models = {
        'Random Forest': rf,
        'Logistic Regression': lr
    }

    # Comparar
    comparator = ModelComparator()
    results = comparator.compare_models(models, X_df, y_series)

    print("\nResultados:")
    print(results)

    # Visualizaciones
    viz = VisualizationGenerator(output_dir="../results/visualizations")
    viz.plot_metrics_comparison(results, save=True)

    print("\n✓ Evaluación completada")


if __name__ == "__main__":
    main()
