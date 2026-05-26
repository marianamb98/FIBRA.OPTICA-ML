import streamlit as st
import joblib
import numpy as np
import os

# 1. Configuración visual de la ventana del navegador
st.set_page_config(page_title="Predicción de Fibra Óptica - Argentina", page_icon="🌐", layout="centered")

st.title("🌐 Clasificador de Fibra Óptica")
st.write("Bienvenido al sistema predictivo desarrollado para el Trabajo Integrador Final de Modelado y Ciencia de Datos.")
st.markdown("---")

# 2. Mecanismo de Almacenamiento / Persistencia (Criterio 3 de la Rúbrica)
# Definimos las rutas a los artefactos binarios que generaste en el Ciclo 2
PATH_MODELO = "models/modelo_fibra_optica.pkl"
PATH_SCALER = "models/scaler_fibra_optica.pkl"

# Control de errores para verificar la existencia del núcleo matemático
if not os.path.exists(PATH_MODELO) or not os.path.exists(PATH_SCALER):
    st.error("❌ ERROR: No se detectaron los archivos binarios (.pkl) en la carpeta /models.")
    st.warning("Por favor, asegúrese de ejecutar primero el script 'src/model_training.py' para entrenar y guardar el núcleo técnico.")
else:
    # Carga persistente del cerebro del modelo y el transformador de datos
    model = joblib.load(PATH_MODELO)
    scaler = joblib.load(PATH_SCALER)

    # 3. Interfaz Gráfica e Interacción Dinámica (Criterio 1 de la Rúbrica)
    st.subheader("📊 Ingrese los Atributos de la Localidad a Evaluar")
    
    # Formulario con rangos geográficos reales de la República Argentina
    latitud = st.number_input("Centroide Latitud (Coordenada Geográfica)", min_value=-55.0, max_value=-20.0, value=-24.78, step=0.01)
    longitud = st.number_input("Centroide Longitud (Coordenada Geográfica)", min_value=-75.0, max_value=-53.0, value=-65.41, step=0.01)
    poblacion = st.number_input("Población Estimada (Variable Demográfica)", min_value=1, max_value=20000000, value=50000, step=500)

    st.markdown("---")

    # 4. Botón de Inferencia en Tiempo Real
    if st.button("⚡ Ejecutar Inferencia Técnica"):
        # Estructuramos la entrada del usuario como una matriz fila
        datos_entrada = np.array([[latitud, longitud, poblacion]])
        
        # Escalamos los datos de entrada con los parámetros guardados del entrenamiento
        datos_escalados = scaler.transform(datos_entrada)
        
        # Realizamos la predicción final
        prediccion = model.predict(datos_escalados)[0]
        
        # CLASIFICADOR PROBABILÍSTICO (Criterio 2 de la Rúbrica)
        # Extraemos el vector con las probabilidades asignadas por Random Forest
        probabilidades = model.predict_proba(datos_escalados)[0]
        
        st.subheader("Resultado del Análisis:")
        
        # Mostramos los resultados adaptando la respuesta visual según la inferencia
        if prediccion == 1:
            porcentaje_certeza = probabilidades[1] * 100
            st.success(f"🟢 **ALTA FACTIBILIDAD:** Las características del entorno indican una alta probabilidad de acceso a la Red de Fibra Óptica.")
            st.metric(label="Certeza del Clasificador Probabilístico", value=f"{porcentaje_certeza:.2f}%")
        else:
            porcentaje_riesgo = probabilidades[0] * 100
            st.warning(f"🔴 **BAJA FACTIBILIDAD:** Los parámetros geodemográficos sugieren vulnerabilidad técnica o falta de infraestructura física activa.")
            st.metric(label="Probabilidad de Aislamiento Digital", value=f"{porcentaje_riesgo:.2f}%")

st.markdown("---")
st.caption("Sistema de Ciencia de Datos Aplicada — Cátedra de Modelado y Ciencia de Datos")