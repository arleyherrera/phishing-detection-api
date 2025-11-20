"""
run_pipeline.py

Script para ejecutar todo el pipeline de ML de detección de phishing.
Ejecuta todos los pasos en orden: recolección, exploración, preprocesamiento,
entrenamiento, ensemble y validación.

Autor: Sistema de Detección de Phishing
Fecha: 2024-2025
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append('src')

# Importar módulos
from data_loader import DataLoader
from preprocessing import Preprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainingManager
from ensemble_models import EnsembleManager
from evaluation import ModelEvaluator, ModelComparator, VisualizationGenerator, ReportGenerator

# Configuración
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Directorios
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Crear directorios si no existen
os.makedirs(f'{RESULTS_DIR}/metrics', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/visualizations', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/reports', exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def print_section(title):
    """Imprime sección con formato."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def step_1_load_data():
    """Paso 1: Cargar datos."""
    print_section("PASO 1/6: CARGA DE DATOS")

    loader = DataLoader(data_dir=DATA_DIR)

    print("Cargando dataset UCI + datos externos (PhishTank + Perú)...")
    X, y, info = loader.load_all_data(
        include_external=True,
        force_download_uci=False
    )

    print(f"\n[OK] Datos cargados exitosamente")
    print(f"  - Total de muestras: {len(X):,}")
    print(f"  - Caracteristicas: {X.shape[1]}")
    print(f"  - Phishing: {(y == 1).sum():,}")
    print(f"  - Legitimas: {(y == 0).sum():,}")

    return X, y, info


def step_2_exploratory_analysis(X, y):
    """Paso 2: Análisis exploratorio."""
    print_section("PASO 2/6: ANÁLISIS EXPLORATORIO")

    print("Generando estadísticas descriptivas...")

    # Seleccionar solo columnas numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]

    # Estadísticas básicas
    print("\nEstadísticas descriptivas:")
    print(X_numeric.describe())

    # Distribución de clases
    print("\nDistribución de clases:")
    print(f"  Phishing (1): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.2f}%)")
    print(f"  Legítimas (0): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.2f}%)")

    # Visualización de correlación (top features)
    print("\nGenerando heatmap de correlación...")
    viz_gen = VisualizationGenerator(output_dir=f'{RESULTS_DIR}/visualizations', dpi=300)

    # Seleccionar top 20 features por varianza
    variances = X_numeric.var()
    top_features = variances.nlargest(20).index.tolist()

    plt.figure(figsize=(14, 12))
    sns.heatmap(X_numeric[top_features].corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Heatmap de Correlación - Top 20 Features por Varianza', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Heatmap guardado: {RESULTS_DIR}/visualizations/correlation_heatmap.png")

    return X_numeric


def step_3_preprocessing(X, y):
    """Paso 3: Preprocesamiento."""
    print_section("PASO 3/6: PREPROCESAMIENTO")

    preprocessor = Preprocessor(random_state=RANDOM_STATE)

    print("Ejecutando pipeline de preprocesamiento...")
    print("  - Limpieza de datos")
    print("  - Selección de features")
    print("  - Normalización (StandardScaler)")
    print("  - Balanceo (SMOTE)")
    print("  - Split 70/15/15 (train/val/test)")

    # Ejecutar pipeline de preprocesamiento
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        X, y,
        apply_balancing=True,
        apply_scaling=True
    )

    # Obtener scaler y features
    scaler = preprocessor.scaler.scaler
    selected_features = X_train.columns.tolist()

    print(f"\n[OK] Preprocesamiento completado")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Val: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")
    print(f"  - Features seleccionadas: {len(selected_features)}")

    # Guardar datos procesados
    os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
    joblib.dump(X_train, f'{DATA_DIR}/processed/X_train.pkl')
    joblib.dump(X_val, f'{DATA_DIR}/processed/X_val.pkl')
    joblib.dump(X_test, f'{DATA_DIR}/processed/X_test.pkl')
    joblib.dump(y_train, f'{DATA_DIR}/processed/y_train.pkl')
    joblib.dump(y_val, f'{DATA_DIR}/processed/y_val.pkl')
    joblib.dump(y_test, f'{DATA_DIR}/processed/y_test.pkl')
    joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
    joblib.dump(selected_features, f'{DATA_DIR}/processed/selected_features.pkl')

    print(f"[OK] Datos guardados en {DATA_DIR}/processed/")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, selected_features


def step_4_train_models(X_train, y_train, X_val, y_val):
    """Paso 4: Entrenamiento de modelos."""
    print_section("PASO 4/6: ENTRENAMIENTO DE MODELOS")

    manager = ModelTrainingManager(random_state=RANDOM_STATE)

    print("Entrenando 8 modelos con hyperparameter tuning...")
    print("  1. Logistic Regression")
    print("  2. Decision Tree")
    print("  3. Random Forest")
    print("  4. XGBoost")
    print("  5. Gradient Boosting")
    print("  6. SVM")
    print("  7. KNN")
    print("  8. Naive Bayes")
    print("\n[!] Esto puede tomar 15-30 minutos...\n")

    trainers = manager.train_all_models(X_train, y_train, use_grid_search=True)

    # Evaluar en validation set
    comparator = ModelComparator()
    results_val = comparator.compare_models(trainers, X_val, y_val, include_time=True)

    print("\n[OK] Entrenamiento completado")
    print("\nResultados en Validation Set:")
    print(results_val[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))

    # Guardar modelos
    manager.save_all_models(trainers, output_dir=MODELS_DIR)
    print(f"\n[OK] Modelos guardados en {MODELS_DIR}/")

    # Guardar resultados
    results_val.to_csv(f'{RESULTS_DIR}/metrics/model_comparison.csv', index=False)

    # Mejor modelo
    best_idx = results_val['f1_score'].idxmax()
    best_model_name = results_val.loc[best_idx, 'model_name']

    with open(f'{RESULTS_DIR}/metrics/best_model.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"F1-Score: {results_val.loc[best_idx, 'f1_score']:.4f}\n")

    print(f"\n[*] Mejor modelo: {best_model_name}")

    return trainers, results_val, best_model_name


def step_5_ensemble_methods(trainers, X_train, y_train, X_val, y_val, results_val):
    """Paso 5: Métodos ensemble."""
    print_section("PASO 5/6: MÉTODOS ENSEMBLE")

    ensemble_manager = EnsembleManager(random_state=RANDOM_STATE)

    # Cargar modelos desde archivos
    loaded_trainers = {}
    for name in trainers.keys():
        model = joblib.load(f'{MODELS_DIR}/{name}.pkl')

        class LoadedTrainer:
            def __init__(self, model, name):
                self.model = model
                self.model_name = name

        loaded_trainers[name] = LoadedTrainer(model, name)

    print("Creando Voting Classifier (soft voting)...")
    voting_ensemble = ensemble_manager.create_voting_ensemble(loaded_trainers, voting='soft')
    voting_ensemble.train(X_train, y_train, cv=5)

    print("Creando Stacking Classifier...")
    stacking_ensemble = ensemble_manager.create_stacking_ensemble(loaded_trainers)
    stacking_ensemble.train(X_train, y_train)

    # Evaluar
    evaluator = ModelEvaluator()
    voting_metrics = evaluator.evaluate_model(voting_ensemble.model, X_val, y_val, model_name="Voting Classifier")
    stacking_metrics = evaluator.evaluate_model(stacking_ensemble.model, X_val, y_val, model_name="Stacking Classifier")

    print("\n[OK] Ensembles creados")
    print(f"\nVoting Classifier:")
    print(f"  - Accuracy: {voting_metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {voting_metrics['f1_score']:.4f}")
    print(f"\nStacking Classifier:")
    print(f"  - Accuracy: {stacking_metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {stacking_metrics['f1_score']:.4f}")

    # Guardar ensembles
    voting_ensemble.save_model(f'{MODELS_DIR}/voting_classifier.pkl')
    stacking_ensemble.save_model(f'{MODELS_DIR}/stacking_classifier.pkl')

    # Comparación final
    ensemble_results = pd.DataFrame([voting_metrics, stacking_metrics])
    all_results = pd.concat([results_val, ensemble_results], ignore_index=True)
    all_results = all_results.sort_values('f1_score', ascending=False)

    print("\n[OK] Comparación completa:")
    print(all_results[['model_name', 'accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

    return all_results


def step_6_final_evaluation(X_test, y_test):
    """Paso 6: Evaluación final."""
    print_section("PASO 6/6: EVALUACIÓN FINAL")

    # Cargar mejor modelo
    with open(f'{RESULTS_DIR}/metrics/best_model.txt', 'r') as f:
        best_model_name = f.readline().replace('Best Model: ', '').strip()

    print(f"Evaluando modelo final en Test Set: {best_model_name}")

    model = joblib.load(f'{MODELS_DIR}/{best_model_name}.pkl')

    evaluator = ModelEvaluator()
    test_metrics = evaluator.evaluate_model(model, X_test, y_test, model_name=best_model_name)

    print(f"\n[OK] Resultados en Test Set:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")
    print(f"  - F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Generar visualizaciones finales
    print("\nGenerando visualizaciones finales...")
    viz_gen = VisualizationGenerator(output_dir=f'{RESULTS_DIR}/visualizations', dpi=300)

    y_pred = model.predict(X_test)
    viz_gen.plot_confusion_matrix(y_test, y_pred, model_name=best_model_name, save=True)

    print(f"[OK] Visualizaciones guardadas en {RESULTS_DIR}/visualizations/")

    return test_metrics


def main():
    """Función principal."""
    print("\n" + "=" * 70)
    print("  SISTEMA DE DETECCIÓN DE PHISHING - PIPELINE COMPLETO")
    print("=" * 70)
    print("\nEjecutando pipeline completo de ML...")
    print("Tiempo estimado: 45-90 minutos\n")

    try:
        # Paso 1: Cargar datos
        X, y, info = step_1_load_data()

        # Paso 2: Análisis exploratorio
        X_numeric = step_2_exploratory_analysis(X, y)

        # Paso 3: Preprocesamiento
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, selected_features = step_3_preprocessing(X_numeric, y)

        # Paso 4: Entrenamiento de modelos
        trainers, results_val, best_model_name = step_4_train_models(X_train, y_train, X_val, y_val)

        # Paso 5: Ensemble methods
        all_results = step_5_ensemble_methods(trainers, X_train, y_train, X_val, y_val, results_val)

        # Paso 6: Evaluación final
        test_metrics = step_6_final_evaluation(X_test, y_test)

        # Resumen final
        print_section("RESUMEN FINAL")
        print("[OK] Pipeline completado exitosamente\n")
        print(f"[DATA] Archivos generados:")
        print(f"  - Modelos: {MODELS_DIR}/ (10 modelos)")
        print(f"  - Datos procesados: {DATA_DIR}/processed/")
        print(f"  - Métricas: {RESULTS_DIR}/metrics/")
        print(f"  - Visualizaciones: {RESULTS_DIR}/visualizations/")
        print(f"\n[*] Mejor modelo: {best_model_name}")
        print(f"  - F1-Score (Test): {test_metrics['f1_score']:.4f}")
        print(f"  - Accuracy (Test): {test_metrics['accuracy']:.4f}")
        print("\n[OK] Sistema listo para produccion")
        print(f"\n[NOTE] Usa 'python predict_url.py <URL>' para analizar URLs")
        print("=" * 70)

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
