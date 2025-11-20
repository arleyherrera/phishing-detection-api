# 🛡️ Sistema de Detección de Phishing con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)](https://scikit-learn.org/)

## 📋 Descripción del Proyecto

Proyecto de tesis universitaria para la detección automática de sitios web de phishing utilizando técnicas de Machine Learning. El sistema analiza características de URLs y predice si un sitio web es legítimo o fraudulento (phishing).

### 🎯 Objetivos

- Desarrollar un modelo de ML robusto para detectar sitios de phishing
- Comparar el rendimiento de múltiples algoritmos de clasificación
- Implementar técnicas de ensemble learning para mejorar la precisión
- Validar el modelo con datos actuales de PhishTank y URLs peruanas
- Generar análisis exhaustivo con visualizaciones profesionales

## 📊 Dataset

### Dataset Principal: PhiUSIIL (UCI Machine Learning Repository)
- **Fuente**: [UCI ML Repository - ID: 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Tamaño**: ~235,000 URLs
  - 134,850 URLs legítimas
  - 100,945 URLs de phishing
- **Features**: 56 características extraídas de las URLs

### Datos Adicionales
- **PhishTank**: URLs de phishing actualizadas y verificadas
- **URLs Legítimas de Perú**: Sitios oficiales de:
  - Bancos (BCP, Interbank, BBVA, Scotiabank, BanBif)
  - Gobierno (gob.pe, SUNAT, RENIEC, MINSA)
  - Universidades (PUCP, UNI, UNMSM, UPC, ULIMA)
  - E-commerce (Falabella, Ripley, Plaza Vea)
  - Medios de comunicación (El Comercio, RPP, La República)

## 🤖 Modelos Implementados

### Modelos Individuales
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**
5. **Gradient Boosting**
6. **Support Vector Machine (SVM)**
7. **K-Nearest Neighbors (KNN)**
8. **Naive Bayes**

### Ensemble Methods
9. **Voting Classifier** (soft voting)
10. **Stacking Classifier**

## 📁 Estructura del Proyecto

```
phishing_detection/
├── data/                              # Datasets
│   ├── raw/                          # Datos sin procesar
│   ├── processed/                    # Datos procesados
│   └── external/                     # Datos de PhishTank y URLs Perú
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 00_data_collection.ipynb      # Recolección de datos
│   ├── 01_data_exploration.ipynb     # Análisis exploratorio
│   ├── 02_preprocessing.ipynb        # Preprocesamiento
│   ├── 03_model_training.ipynb       # Entrenamiento de modelos
│   ├── 04_ensemble_methods.ipynb     # Métodos ensemble
│   └── 05_peru_validation.ipynb      # Validación con datos peruanos
│
├── src/                               # Código fuente
│   ├── data_collection.py            # Recolección de PhishTank y URLs Perú
│   ├── feature_extraction.py         # Extracción de 56 features
│   ├── data_loader.py                # Carga de datasets
│   ├── preprocessing.py              # Preprocesamiento de datos
│   ├── feature_engineering.py        # Ingeniería de características
│   ├── model_training.py             # Entrenamiento de modelos
│   ├── ensemble_models.py            # Modelos ensemble
│   └── evaluation.py                 # Evaluación y métricas
│
├── models/                            # Modelos entrenados (.pkl)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── ...
│
├── results/                           # Resultados
│   ├── metrics/                      # Métricas en CSV
│   ├── visualizations/               # Gráficas (300 DPI)
│   └── reports/                      # Reportes finales
│
├── requirements.txt                   # Dependencias
└── README.md                         # Este archivo
```

## 🚀 Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd phishing_detection
```

2. **Crear entorno virtual** (recomendado)
```bash
python -m venv venv
```

3. **Activar entorno virtual**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

5. **Verificar instalación**
```bash
pip list
```

## 💻 Uso

### 1. Recolección de Datos
```bash
jupyter notebook notebooks/00_data_collection.ipynb
```
Descarga datos de PhishTank, recolecta URLs peruanas y extrae features.

### 2. Exploración de Datos
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Análisis exploratorio completo del dataset.

### 3. Preprocesamiento
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```
Limpieza, normalización y preparación de datos.

### 4. Entrenamiento de Modelos
```bash
jupyter notebook notebooks/03_model_training.ipynb
```
Entrena los 8 modelos individuales con hyperparameter tuning.

### 5. Ensemble Methods
```bash
jupyter notebook notebooks/04_ensemble_methods.ipynb
```
Implementa y evalúa Voting y Stacking classifiers.

### 6. Validación con Datos Peruanos
```bash
jupyter notebook notebooks/05_peru_validation.ipynb
```
Evalúa el rendimiento en URLs específicas de Perú.

## 📈 Métricas de Evaluación

Para cada modelo se calculan las siguientes métricas:

- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De los clasificados como phishing, ¿cuántos son reales?
- **Recall**: De todos los phishing reales, ¿cuántos se detectaron?
- **F1-Score**: Media armónica entre precision y recall
- **ROC-AUC**: Área bajo la curva ROC
- **Confusion Matrix**: Matriz de confusión
- **Tiempo de entrenamiento**
- **Tiempo de inferencia**

## 📊 Visualizaciones

El proyecto genera las siguientes visualizaciones profesionales (300 DPI):

- Distribución de clases
- Matriz de correlación de features
- Importancia de características
- Curvas ROC de todos los modelos
- Matrices de confusión
- Comparación de métricas entre modelos
- Learning curves
- Análisis de errores

## 🛠️ Buenas Prácticas Implementadas

✅ **Código modular y reutilizable** (PEP 8)
✅ **Docstrings** en todas las funciones
✅ **random_state=42** para reproducibilidad
✅ **Logging** para seguimiento
✅ **Manejo de excepciones**
✅ **Comentarios claros** y documentación completa
✅ **Gráficas en alta resolución** (300 DPI)
✅ **Cross-validation** (5-fold)
✅ **Hyperparameter tuning** (GridSearchCV/RandomizedSearchCV)
✅ **Split estratificado** (70% train, 15% val, 15% test)

## 🔬 Features Extraídas (56 características)

Las características extraídas de cada URL incluyen:

- Longitud de URL, dominio y path
- Cantidad de caracteres especiales (@, -, _, etc.)
- Presencia de HTTPS/HTTP
- Cantidad de subdominios
- Uso de IP en la URL
- Edad del dominio (WHOIS)
- Presencia de palabras sospechosas
- Entropía de la URL
- Y 45+ características adicionales

## 📝 Resultados

Los resultados detallados se encuentran en:
- `results/metrics/model_comparison.csv` - Comparación de todos los modelos
- `results/reports/final_report.md` - Reporte final completo
- `results/visualizations/` - Todas las gráficas generadas

## 🎓 Autores

**Tu Nombre**
Universidad [Nombre de tu Universidad]
Tesis de [Carrera]
Año: 2024-2025

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- UCI Machine Learning Repository por el dataset PhiUSIIL
- PhishTank por proporcionar datos actualizados de phishing
- Comunidad de scikit-learn y XGBoost
- Profesores y asesores de tesis

## 📧 Contacto

Para preguntas o sugerencias:
- Email: tu-email@universidad.edu.pe
- LinkedIn: [Tu perfil]

---

**Nota**: Este proyecto es parte de una tesis universitaria y tiene fines educativos y de investigación.

🛡️ **¡Juntos contra el phishing!**
