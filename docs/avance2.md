# Avance 2: Núcleo Técnico del Sistema Funcionando
**Materia:** Modelado y Ciencia de Datos  
**Ciclo 2:** Machine Learning Clásico Supervisado — Clasificador Local

---

## 1. Arquitectura Centralizada del Núcleo Técnico
Siguiendo las decisiones del diseño original, el sistema opera bajo una **arquitectura centralizada**. Los tres archivos de datos origen (`localidades.csv`, `poblacion.csv` y `nodos_fibra.csv`) han sido depurados, decodificados usando la codificación correcta para caracteres en español (`latin-1`) y unificados localmente. El procesamiento, entrenamiento y evaluación se ejecutan completamente en un único entorno local controlado.

## 2. Estructura del Repositorio Utilizada
Para garantizar la modularidad y reproducibilidad del proyecto, replicamos la estructura de trabajo distribuida de la siguiente manera:
* `data/raw/`: Archivos fuentes descargados de Datos Argentina y SimpleMaps.
* `data/processed/`: Contiene el dataset final unificado `features_localidades.csv`.
* `src/data_processing.py`: Script encargado de la carga, limpieza y partición estratificada de datos (80% entrenamiento, 20% test).
* `src/model_training.py`: Script ejecutable que entrena el algoritmo, calcula las métricas y exporta el modelo.
* `models/`: Carpeta donde se almacena el "cerebro" del sistema en formato binario (`.pkl`).

---

## 3. Evaluación del Modelo y Métricas Obtenidas
Al ejecutar nuestro núcleo técnico entrenado mediante un clasificador de **Random Forest (Bosques Aleatorios)**, obtuvimos los siguientes resultados en el conjunto de test independiente:

* **Accuracy General:** 0.7575 (75.75% de predicciones correctas a nivel global).
* **F1-Score (Clase 0 - Sin Fibra Óptica):** 0.85
* **F1-Score (Clase 1 - Con Fibra Óptica):** 0.44

### Reporte de Clasificación Detallado:
```text
              precision    recall  f1-score   support

           0       0.80      0.89      0.85       617
           1       0.55      0.37      0.44       216

    accuracy                           0.76       833