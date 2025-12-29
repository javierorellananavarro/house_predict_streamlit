import streamlit as st
import joblib
import numpy as np

# Configuración general
st.set_page_config(
    page_title="Predicción de Precios de Casas",
    page_icon="🏠",
    layout="centered"
)

# Cargar modelo
model = joblib.load("model.pkl")

# Título principal
st.title("🏠 Predicción del Precio de Casas en la India.")
st.markdown(
    """
    Esta aplicación utiliza **Machine Learning** para estimar el precio de una casa  
    a partir de sus características principales, tales como:
     - Cantidad de Habitaciones
     - Longitud de la Casa en pies cuadrados
     - Cantidad de Baños
     - Condición de la Casa (1 al 5)
     - Escuelas cercanas a la Casa.
    """
)

st.divider()

# Barra lateral para los inputs
st.sidebar.header("📋 Características de la Casa")

habitaciones = st.sidebar.number_input(
    "Número de habitaciones",
    min_value=0,
    value=1,
    help="Cantidad total de habitaciones"
)

banos = st.sidebar.number_input(
    "Número de baños",
    min_value=0,
    value=1,
    help="Cantidad total de baños"
)

area_vivienda = st.sidebar.number_input(
    "Área habitable (pies cuadrados)",
    min_value=0,
    value=2000,
    help="Área total de la vivienda"
)

condicion = st.sidebar.slider(
    "Condición de la vivienda",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Muy mala, 5 = Excelente"
)

escuelas_cercanas = st.sidebar.number_input(
    "Escuelas cercanas",
    min_value=0,
    value=0,
    help="Número de escuelas en la zona"
)

# Resumen de entradas
st.subheader("📊 Resumen de Datos Ingresados")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Habitaciones:** {habitaciones}")
    st.write(f"**Baños:** {banos}")
    st.write(f"**Área:** {area_vivienda} ft²")

with col2:
    st.write(f"**Condición:** {condicion}")
    st.write(f"**Escuelas cercanas:** {escuelas_cercanas}")

st.divider()

# Botón de predicción
boton_predecir = st.button("🔍 Predecir Precio de una casa!", use_container_width=True)

if boton_predecir:
    X = np.array([[habitaciones, banos, area_vivienda, condicion, escuelas_cercanas]])
    prediccion = model.predict(X)[0]

    st.success(f"💰 **Precio estimado de la vivienda:** ${prediccion:,.2f}")

else:
    st.info("👈 Ingresa los datos y presiona **Predecir Precio**")
