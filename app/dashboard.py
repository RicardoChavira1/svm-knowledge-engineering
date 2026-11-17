import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Analítica de Riesgo Bancario | SVM México",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (Estética Banca Corporativa / FinTech)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131720, #191F2C);
        border: 1px solid #2A3241;
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .banner-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        padding: 20px 25px;
        border-radius: 10px;
        margin-bottom: 22px;
    }
    .banner-title {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    .banner-sub {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-mx {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 10px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)

# Rutas de los artefactos serializados por el equipo
MODEL_PATH = "models/svm_model.joblib"
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/encoder.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

# Encabezado ejecutivo
st.markdown("""
    <div class="banner-header">
        <p class="banner-title">Sistema de Scoring y Retención de Clientes <span class="badge-mx">MXN • SVM RBF</span></p>
        <p class="banner-sub">Ingeniería del Conocimiento • TecNM Campus Toluca • Evaluación Geométrica de Fronteras de Decisión</p>
    </div>
""", unsafe_allow_html=True)

# Verificación de modelos entrenados
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
    st.error("No se detectaron los archivos entrenados en `models/`. Ejecuta `py src/train.py` primero.")
    st.stop()

model, scaler, encoder = load_artifacts()

# Diccionario de homologación de países (Permite buscar cualquier país escribiendo y lo mapea al encoder entrenado)
PAISES_MAP = {
    "México": "Spain",
    "España": "Spain",
    "Francia": "France",
    "Alemania": "Germany",
    "Estados Unidos": "Germany",
    "Canadá": "France",
    "Colombia": "Spain",
    "Argentina": "Spain",
    "Chile": "Spain",
    "Brasil": "Spain",
    "Perú": "Spain",
    "Reino Unido": "France",
    "Italia": "Spain",
    "Japón": "Germany"
}

GENEROS_MAP = {
    "Femenino": "Female",
    "Masculino": "Male"
}

TIPO_CAMBIO_MXN = 20.0  # Conversión $20 MXN = 1 USD para compatibilidad con escala del entrenamiento

# --- BARRA LATERAL: PERFIL DEL CLIENTE ---
with st.sidebar:
    st.markdown("### Perfil Demográfico")
    st.caption("Puedes escribir directamente en las listas para buscar.")
    
    pais_es = st.selectbox(
        "País de Residencia (Escribe para buscar)",
        options=list(PAISES_MAP.keys()),
        index=0  # México por defecto
    )
    
    genero_es = st.selectbox(
        "Género",
        options=list(GENEROS_MAP.keys()),
        index=0
    )
    
    edad = st.slider("Edad del Titular (años)", min_value=18, max_value=90, value=38)
    antiguedad = st.slider("Antigüedad en el Banco (años)", min_value=0, max_value=10, value=4)
    
    st.divider()
    st.markdown("### Situación Financiera (MXN)")
    
    credit_score = st.slider(
        "Score Buró de Crédito",
        min_value=300,
        max_value=850,
        value=660,
        step=5,
        help="Puntaje crediticio estándar (300 - 850 pts)."
    )
    
    saldo_mxn = st.number_input(
        "Saldo Promedio en Cuenta ($ MXN)",
        min_value=0.0,
        max_value=5000000.0,
        value=850000.0,
        step=25000.0,
        format="%.2f"
    )
    
    ingreso_mxn = st.number_input(
        "Ingreso Anual Estimado ($ MXN)",
        min_value=0.0,
        max_value=4000000.0,
        value=1200000.0,
        step=25000.0,
        format="%.2f"
    )
    
    productos_label = st.selectbox(
        "Productos Financieros Contratados",
        options=[
            "1 - Cuenta de Débito / Nómina",
            "2 - Débito + Tarjeta de Crédito",
            "3 - Débito + Crédito + Inversión",
            "4 - Portafolio Patrimonial Completo"
        ],
        index=1
    )
    num_productos = int(productos_label.split(" - ")[0])

# Capa de traducción hacia el esquema exacto que espera el encoder de tus compañeros
input_model_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": PAISES_MAP[pais_es],
    "Gender": GENEROS_MAP[genero_es],
    "Age": edad,
    "Tenure": antiguedad,
    "Balance": saldo_mxn / TIPO_CAMBIO_MXN,
    "NumOfProducts": num_productos,
    "EstimatedSalary": ingreso_mxn / TIPO_CAMBIO_MXN
}])

# Ejecución del Pipeline de Inferencia
encoded_input = encoder.transform(input_model_df)
scaled_input = scaler.transform(encoded_input)

prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]
decision_dist = model.decision_function(scaled_input)[0]

# --- CUERPO PRINCIPAL CON PESTAÑAS ---
tab_diagnostico, tab_simulacion = st.tabs(["Diagnóstico Geométrico SVM", "Impacto Financiero y Ficha Técnica"])

with tab_diagnostico:
    # Fila superior de KPIs
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric(
            label="Dictamen del Clasificador",
            value="ALTO RIESGO" if prediction == 1 else "CLIENTE ESTABLE",
            delta="Probable Abandono" if prediction == 1 else "Retención Segura",
            delta_color="inverse" if prediction == 1 else "normal"
        )
    with k2:
        st.metric(
            label="Probabilidad de Fuga",
            value=f"{probability * 100:.2f}%",
            delta=f"{(probability - 0.5) * 100:+.1f}% vs punto de corte",
            delta_color="inverse"
        )
    with k3:
        st.metric(
            label="Distancia al Hiperplano",
            value=f"{decision_dist:+.4f}",
            delta="Lado Positivo (+)" if decision_dist > 0 else "Lado Negativo (-)",
            delta_color="off"
        )
    with k4:
        capital_riesgo = saldo_mxn if prediction == 1 else 0.0
        st.metric(
            label="Capital en Riesgo (MXN)",
            value=f"${capital_riesgo:,.0f}",
            delta="Requiere retención" if prediction == 1 else "Capital protegido",
            delta_color="inverse" if prediction == 1 else "normal"
        )

    st.markdown("#### Proyección en el Espacio de Decisión ($w^T x + b$)")
    
    col_grafica, col_interpretacion = st.columns([1.7, 1], gap="large")
    
    with col_grafica:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#131720')

        # Regiones de clasificación
        ax.axvspan(-3.5, -1.0, color='#10B981', alpha=0.12, label='Retención Fuerte (< -1.0)')
        ax.axvspan(-1.0, 1.0, color='#F59E0B', alpha=0.12, label='Margen de Soporte [-1.0, +1.0]')
        ax.axvspan(1.0, 3.5, color='#EF4444', alpha=0.14, label='Riesgo de Fuga (> +1.0)')

        # Fronteras
        ax.axvline(0, color='#F8FAFC', linestyle='-', linewidth=2.0, label='Hiperplano Óptimo (0.0)')
        ax.axvline(-1, color='#64748B', linestyle='--', linewidth=1.1)
        ax.axvline(1, color='#64748B', linestyle='--', linewidth=1.1)

        # Marcador del cliente
        plot_x = np.clip(decision_dist, -3.3, 3.3)
        color_punto = '#EF4444' if prediction == 1 else '#10B981'
        ax.scatter([plot_x], [0], color=color_punto, s=200, zorder=6, edgecolors='#FFFFFF', linewidth=2)
        
        ax.annotate(
            f"Cliente Evaluado\n({decision_dist:+.3f})",
            xy=(plot_x, 0),
            xytext=(plot_x, 0.30),
            ha='center',
            color='#F8FAFC',
            fontsize=9.5,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec=color_punto, lw=1.2)
        )

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.55, 0.72)
        ax.set_yticks([])
        ax.set_xlabel("Distancia Orientada Normalizada al Hiperplano", color='#94A3B8', fontsize=9)
        ax.tick_params(axis='x', colors='#94A3B8', labelsize=8.5)
        
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        
        ax.legend(loc='upper right', frameon=True, facecolor='#0F172A', edgecolor='#334155', fontsize=8, labelcolor='#E2E8F0', ncol=2)
        st.pyplot(fig, use_container_width=True)

    with col_interpretacion:
        st.markdown("**Lectura Técnica del Resultado**")
        if abs(decision_dist) < 1.0:
            st.warning(
                f"**Zona de Incertidumbre (Dentro del Margen):**  \n"
                f"La distancia `{decision_dist:+.4f}` cayó dentro de la avenida de seguridad `[-1, +1]`. "
                f"Este perfil se comporta como un **Vector de Soporte** cercano a la frontera."
            )
        elif decision_dist >= 1.0:
            st.error(
                f"**Abandono Confirmado (Fuera del Margen):**  \n"
                f"Con `{decision_dist:+.4f}`, el cliente superó la banqueta positiva (`+1.0`). "
                f"Se recomienda activar campaña de retención inmediata."
            )
        else:
            st.success(
                f"**Fidelidad Sólida (Fuera del Margen):**  \n"
                f"Con `{decision_dist:+.4f}`, el cliente se ubica lejos de la frontera de riesgo (`< -1.0`)."
            )
        
        st.caption(
            f"Nota de homologación: El perfil de **{pais_es}** ({genero_es}) fue normalizado en pesos mexicanos "
            f"y proyectado mediante el Kernel RBF entrenado."
        )

with tab_simulacion:
    st.markdown("#### Auditoría del Vector de Características (Español vs Esquema del Modelo)")
    
    c_tabla1, c_tabla2 = st.columns(2, gap="large")
    
    with c_tabla1:
        st.markdown("**1. Datos Capturados en Interfaz (México)**")
        df_visual = pd.DataFrame({
            "Variable": ["País", "Género", "Edad", "Antigüedad", "Score Crediticio", "Saldo en Cuenta", "Ingreso Anual", "Productos"],
            "Valor Registrado": [
                pais_es,
                genero_es,
                f"{edad} años",
                f"{antiguedad} años",
                f"{credit_score} pts",
                f"${saldo_mxn:,.2f} MXN",
                f"${ingreso_mxn:,.2f} MXN",
                f"{num_productos} productos"
            ]
        })
        st.dataframe(df_visual, use_container_width=True, hide_index=True)

    with c_tabla2:
        st.markdown("**2. Vector Transformado para `svm_model.joblib`**")
        df_tecnico = input_model_df.T.rename(columns={0: "Valor de Entrada al Pipeline"})
        st.dataframe(df_tecnico, use_container_width=True)