**CICLO 1: Machine Learning Clasico supervisado - Clasificador local** *Datos csv, EDA y procesamiento (Pandas, Matplotlib), Modelo Supervisado, Clasificador con probabilidad, App local (Streamlit o FastAPI)*

* **Nombre del proyecto:** *Predicción de conectividad a fibra óptica en localidades argentinas*
* **Problema:** *Existe desigualdad en el acceso a internet de alta velocidad en distintas regiones del país*
* **Solución:** *Desarrollar un modelo de Machine Learning que permita predecir si una localidad tiene acceso a la red de fibra óptica en función de variables geográficas y demográficas*

**DATASET: Se unieron 3 archivos csv para poder trabajar con el modelo de clasificacion.**

* Primero de la pagina de Datos Argentina se recolectaron los datos de las localidades en el archivo localidades.csv, cada una con sus respectivas coordenadas de latitud y longitud en la superficie terrestre, su nombre de departamento y provincia a la que pertenece. **Servicio de normalizacion de direcciones y unidades territoriales de Argentina url:** https://datos.gob.ar/dataset/jgm-servicio-normalizacion-direcciones-unidades-territoriales-argentina

* Los datos de población provienen de una fuente secundaria (SimpleMaps), basada en estimaciones y registros históricos. Si bien pueden no reflejar valores exactos del último censo, resultan adecuados para el modelado exploratorio y el desarrollo del sistema predictivo. Se priorizó un dataset limpio y estructurado para facilitar el procesamiento y modelado, sacrificando precisión censal absoluta en favor de consistencia y usabilidad. url: https://simplemaps.com/data/ar-cities.csv?utm_source=chatgpt.com

* Los datos de nodos de la Red Federal de Fibra Optica tambien de la pagina de Datos Argentina url: https://datos.gob.ar/dataset/arsat-puntos-conexion-refefo

**Link al repositorio GitHub:** https://github.com/Vixxtory/fibra-optica-ml-TIF/tree/main/fibra-optica-ml

**STACK:** *Python, Pandas, Scikit-learn, QGIS, Streamlit*

**ARQUITECTURA DEL MODELO:** *CENTRALIZADA*