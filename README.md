# Wine Quality Classification Analysis

## Objetivo

Utilizar tecnicas de clasificacion aprendidas hasta el momento para predecir la calidad del vino basandose en caracteristicas fisico-quimicas. Este ejercicio permite aplicar conceptos como la seleccion de caracteristicas, preprocesamiento de datos, entrenamiento y evaluacion de modelos de clasificacion, y analisis de resultados mediante metricas y visualizaciones.

## Dataset: Wine Quality Dataset

**Descripcion del Dataset:** Este conjunto de datos contiene informacion sobre distintas caracteristicas fisico-quimicas de muestras de vino tinto y su calidad asociada. Las caracteristicas incluyen:

- **fixed acidity**: Acidez fija
- **volatile acidity**: Acidez volatil
- **citric acid**: Acido citrico
- **residual sugar**: Azucar residual
- **chlorides**: Cloruros
- **free sulfur dioxide**: Dioxido de azufre libre
- **total sulfur dioxide**: Dioxido de azufre total
- **density**: Densidad
- **pH**: pH
- **sulphates**: Sulfatos
- **alcohol**: Alcohol
- **quality**: Calidad del vino (escala 0-10)

## Estructura del Proyecto

```
Core4_WineDs/
├── WineQT.csv                    # Dataset original
├── wine_quality_classification.ipynb  # Notebook principal
├── README.md                     # Este archivo
└── requirements.txt              # Dependencias del proyecto
```

## Tecnicas Utilizadas

### 1. Carga y Exploracion de Datos
- Analisis de estructura basica del dataset
- Descripcion de variables y distribuciones
- Identificacion y tratamiento de valores nulos y outliers
- Analisis de correlaciones entre variables

### 2. Preprocesamiento de Datos
- Seleccion de caracteristicas importantes usando SelectKBest
- Transformacion de variables categoricas
- Division de datos en conjuntos de entrenamiento y prueba
- Escalado de caracteristicas usando StandardScaler
- Binarizacion de la variable objetivo (calidad)

### 3. Entrenamiento de Modelos
- **K-Nearest Neighbors (KNN)**: Clasificacion basada en vecinos cercanos
- **Random Forest**: Ensemble de arboles de decision
- **Regresion Logistica**: Clasificacion lineal con regularizacion
- Validacion cruzada para optimizacion de hiperparametros
- GridSearch para busqueda de mejores parametros

### 4. Evaluacion de Modelos
- Metricas de evaluacion:
  - Accuracy (Exactitud)
  - Precision
  - Recall
  - F1-Score
  - Matriz de confusion
  - Curva ROC y AUC

### 5. Analisis de Resultados
- Comparacion de rendimiento entre modelos
- Analisis de importancia de caracteristicas
- Visualizaciones de resultados
- Conclusiones y recomendaciones

## Como Ejecutar el Codigo

### Prerrequisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Instrucciones

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   cd Core4_WineDs
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el notebook:**
   ```bash
   jupyter notebook wine_quality_classification.ipynb
   ```

4. **Ejecutar todas las celdas del notebook** en orden secuencial.

## Resultados Principales

### Modelos Evaluados
1. **K-Nearest Neighbors (KNN)**
   - Ventajas: Simple, facil de interpretar
   - Desventajas: Sensible a dimensionalidad y outliers

2. **Random Forest**
   - Ventajas: Maneja datos no lineales, proporciona importancia de caracteristicas
   - Desventajas: Menos interpretable, computacionalmente intensivo

3. **Regresion Logistica**
   - Ventajas: Muy interpretable, eficiente computacionalmente
   - Desventajas: Asume relacion lineal

### Caracteristicas Mas Importantes
Basado en el analisis de Random Forest, las caracteristicas mas importantes para predecir la calidad del vino son:
1. Alcohol
2. Volatile acidity
3. Sulphates
4. Total sulfur dioxide
5. Density

## Archivos Generados

Despues de ejecutar el notebook, se generaran los siguientes archivos:
- `best_wine_quality_model.pkl`: Mejor modelo entrenado
- `wine_quality_scaler.pkl`: Escalador de caracteristicas
- `wine_quality_label_encoder.pkl`: Codificador de etiquetas
- `selected_features.pkl`: Lista de caracteristicas seleccionadas

## Uso del Modelo

```python
import joblib
import pandas as pd

# Cargar componentes del modelo
model = joblib.load('best_wine_quality_model.pkl')
scaler = joblib.load('wine_quality_scaler.pkl')
le = joblib.load('wine_quality_label_encoder.pkl')
selected_features = joblib.load('selected_features.pkl')

# Ejemplo de prediccion
wine_features = {
    'fixed acidity': 8.32,
    'volatile acidity': 0.53,
    'citric acid': 0.27,
    'residual sugar': 2.54,
    'chlorides': 0.09,
    'free sulfur dioxide': 15.87,
    'total sulfur dioxide': 46.47,
    'density': 0.996,
    'pH': 3.31,
    'sulphates': 0.66,
    'alcohol': 10.42
}

# Predecir calidad
features_df = pd.DataFrame([wine_features])
features_df = features_df[selected_features]
features_scaled = scaler.transform(features_df)
prediction = model.predict(features_scaled)[0]
predicted_quality = le.inverse_transform([prediction])[0]
print(f"Calidad predicha: {predicted_quality}")
```

## Conclusiones

1. **Rendimiento de Modelos**: El Random Forest mostro el mejor rendimiento general en terminos de F1-Score.

2. **Caracteristicas Clave**: El contenido de alcohol y la acidez volatil son los factores mas importantes para determinar la calidad del vino.

3. **Aplicacion Practica**: El modelo puede ser utilizado en la industria vinicola para evaluar la calidad durante el proceso de produccion.

4. **Mejoras Futuras**: Se recomienda experimentar con tecnicas de balanceo de clases y modelos mas avanzados como XGBoost.

## Autor

[Tu Nombre]

## Licencia

Este proyecto esta bajo la Licencia MIT. Ver el archivo `LICENSE` para mas detalles.

## Version

v1.0.0 - Analisis inicial de clasificacion de calidad de vino 