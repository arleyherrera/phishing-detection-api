# 📊 Reporte Final - Sistema de Detección de Phishing
**Fecha de generación:** 2025-11-19 23:25:39
---

## 🎯 Resumen Ejecutivo

Se entrenaron y evaluaron **6 modelos** de Machine Learning para la detección automática de sitios web de phishing.

**Mejor Modelo:** random_forest

## 📁 Dataset

- **Total de muestras:** 259287
- **Features:** 40
- **Muestras de phishing:** 134850
- **Muestras legítimas:** 124437

## 📈 Resultados de Modelos

### Tabla Comparativa

|   accuracy |   precision |   recall |   f1_score |   roc_auc | model_name          |   training_time |
|-----------:|------------:|---------:|-----------:|----------:|:--------------------|----------------:|
|   1        |    1        |        1 |   1        |         1 | random_forest       |        54.9682  |
|   1        |    1        |        1 |   1        |         1 | xgboost             |         5.98636 |
|   1        |    1        |        1 |   1        |         1 | Voting Classifier   |       nan       |
|   1        |    1        |        1 |   1        |         1 | Stacking Classifier |       nan       |
|   0.999972 |    0.999951 |        1 |   0.999975 |         1 | logistic_regression |        22.4836  |
|   0.999972 |    0.999951 |        1 |   0.999975 |         1 | svm                 |        64.8365  |

### 🏆 Mejor Modelo: random_forest

- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **ROC-AUC:** 1.0000

## 💡 Conclusiones

1. El sistema logró detectar sitios de phishing con alta precisión.
2. El modelo **random_forest** obtuvo el mejor rendimiento general.
3. Las métricas indican un balance adecuado entre precisión y recall.

## 🎓 Recomendaciones

1. Actualizar el modelo periódicamente con nuevos datos de phishing.
2. Monitorear falsos positivos para ajustar el umbral de clasificación.
3. Integrar el modelo en un sistema de detección en tiempo real.

---
*Generado automáticamente por el Sistema de Detección de Phishing*
