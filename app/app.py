import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import folium
from streamlit_folium import st_folium

# 1. Configuración de la página web
st.set_page_config(page_title="Predicción de Fibra Óptica - Argentina", page_icon="🌐", layout="centered")

st.title("🌐 Clasificador Avanzado de Fibra Óptica")
st.write("Sistema predictivo interactivo con captura de coordenadas geoespaciales en tiempo real.")
st.markdown("---")

# 2. Persistencia del Modelo (.pkl)
PATH_MODELO = "models/modelo_fibra_optica.pkl"
PATH_SCALER = "models/scaler_fibra_optica.pkl"

if not os.path.exists(PATH_MODELO) or not os.path.exists(PATH_SCALER):
    st.error("❌ ERROR: No se detectaron los archivos binarios (.pkl) en la carpeta /models.")
else:
    model = joblib.load(PATH_MODELO)
    scaler = joblib.load(PATH_SCALER)

    # 3. INTERFAZ: MAPA INTERACTIVO DE SELECCIÓN (Doble funcionalidad)
    st.subheader("📍 Selección de Localidad en el Mapa")
    st.info("💡 Modo de uso: Hacé un clic en el mapa de Argentina para capturar las coordenadas automáticamente, o escribilas a mano abajo.")

    # Inicializamos las coordenadas por defecto (Salta) en el estado de la aplicación
    if 'lat_defecto' not in st.session_state:
        st.session_state.lat_defecto = -24.78
        st.session_state.lon_defecto = -65.41

    # Creamos el mapa base centrado en Argentina
    mapa_interactivo = folium.Map(location=[st.session_state.lat_defecto, st.session_state.lon_defecto], zoom_start=5)
    
    # Dibujamos UN SOLO marcador dinámico en la posición actual seleccionada
    folium.Marker(
        location=[st.session_state.lat_defecto, st.session_state.lon_defecto],
        popup="Punto Seleccionado",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(mapa_interactivo)
    
    # Renderizamos el mapa y escuchamos los eventos
    output_mapa = st_folium(mapa_interactivo, width=700, height=400, key="mapa_argentina")

    # Si el usuario hace un nuevo clic, actualizamos la posición del único marcador
    if output_mapa and output_mapa.get("last_clicked"):
        nueva_lat = round(output_mapa["last_clicked"]["lat"], 4)
        nueva_lon = round(output_mapa["last_clicked"]["lng"], 4)
        
        if nueva_lat != st.session_state.lat_defecto or nueva_lon != st.session_state.lon_defecto:
            st.session_state.lat_defecto = nueva_lat
            st.session_state.lon_defecto = nueva_lon
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Atributos de la Localidad a Evaluuar")

    # Formulario dinámico conectado al estado del mapa
    latitud = st.number_input("Centroide Latitud", min_value=-55.0, max_value=-20.0, value=float(st.session_state.lat_defecto), step=0.01)
    longitud = st.number_input("Centroide Longitud", min_value=-75.0, max_value=-53.0, value=float(st.session_state.lon_defecto), step=0.01)
    poblacion = st.number_input("Población Estimada (Variable Demográfica)", min_value=1, max_value=20000000, value=50000, step=500)

    st.markdown("---")

    # 4. Botón de Inferencia en Tiempo Real
    if st.button("⚡ Ejecutar Inferencia Técnica"):
        datos_entrada = np.array([[latitud, longitud, poblacion]])
        datos_escalados = scaler.transform(datos_entrada)
        
        prediccion = model.predict(datos_escalados)[0]
        probabilidades = model.predict_proba(datos_escalados)[0]
        
        st.subheader("Resultado del Análisis:")
        
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