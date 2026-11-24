import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

USER_SITE = r"C:\Users\esqui\AppData\Roaming\Python\Python315\site-packages"
if USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

MODELS_DIR = Path(
    r"C:\Users\esqui\OneDrive\Documents\11 SEMESTRE\ING CONOC\Proyecto_Final\svm-knowledge-engineering\models"
)

def get_val(df, keyword, default):
    for col in df.columns:
        if keyword.lower() in col.lower():
            val = df[col].iloc[0]
            if pd.notna(val):
                return val
    return default

edad = float(get_val(dataset, "edad", 52))
saldo_mxn = float(get_val(dataset, "saldo", 2200000))
credit = float(get_val(dataset, "credit", 620))
productos = float(get_val(dataset, "prod", 1))
pais_es = str(get_val(dataset, "pais", "Alemania"))
genero_es = str(get_val(dataset, "gen", "Femenino"))

PAISES_MAP = {"Francia": "France", "Alemania": "Germany", "España": "Spain", "México": "Spain"}
GENEROS_MAP = {"Femenino": "Female", "Masculino": "Male"}

model = joblib.load(MODELS_DIR / "svm_model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
encoder = joblib.load(MODELS_DIR / "encoder.joblib")

input_df = pd.DataFrame([{
    "CreditScore": credit,
    "Geography": PAISES_MAP.get(pais_es, "Germany"),
    "Gender": GENEROS_MAP.get(genero_es, "Female"),
    "Age": edad,
    "Tenure": 4.0,
    "Balance": saldo_mxn / 20.0,
    "NumOfProducts": productos,
    "EstimatedSalary": 1200000.0 / 20.0,
}])

X_enc = np.array(encoder.transform(input_df), dtype=np.float32)
X_sc = scaler.transform(X_enc)

pred = int(model.predict(X_sc)[0])
proba = float(model.predict_proba(X_sc)[0][1]) * 100.0
dist = float(model.decision_function(X_sc)[0])
cap_riesgo = saldo_mxn if pred == 1 else 0.0

if abs(dist) < 1.0:
    color_main, bg_soft = "#D97706", "#FFFBEB"
    titulo = "ZONA DE MARGEN / VECTOR DE SOPORTE"
elif pred == 1:
    color_main, bg_soft = "#DC2626", "#FEF2F2"
    titulo = "ALERTA CRÍTICA: RIESGO DE FUGA (CLASE 1)"
else:
    color_main, bg_soft = "#16A34A", "#F0FDF4"
    titulo = "CLIENTE ESTABLE / FIDELIZADO (CLASE 0)"

fig = plt.figure(figsize=(7.6, 5.1), dpi=130, facecolor="#FFFFFF")

ax1 = fig.add_axes([0.03, 0.56, 0.94, 0.41])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")

ax1.add_patch(FancyBboxPatch((0.0, 0.44), 1.0, 0.54, boxstyle="round,pad=0.01,rounding_size=0.04",
                             facecolor=bg_soft, edgecolor=color_main, linewidth=2.2))
ax1.text(0.03, 0.80, titulo, color=color_main, fontsize=12.5, fontweight="bold", va="center")
ax1.text(
    0.03, 0.58,
    f"Perfil nuevo: {pais_es} | {genero_es} | {int(edad)} años | Score {int(credit)} | "
    f"{int(productos)} prod. | Saldo ${saldo_mxn:,.0f} MXN",
    color="#334155", fontsize=8.8, va="center"
)

kpis = [
    ("PROBABILIDAD FUGA", f"{proba:.2f}%", color_main),
    ("DISTANCIA HIPERPLANO", f"{dist:+.4f}", "#1E293B"),
    ("CAPITAL EN RIESGO", f"${cap_riesgo:,.0f} MXN", "#DC2626" if pred == 1 else "#16A34A"),
]
for i, (lbl, val, col) in enumerate(kpis):
    x0 = i * 0.34
    ax1.add_patch(FancyBboxPatch((x0, 0.02), 0.32, 0.36, boxstyle="round,pad=0.01,rounding_size=0.03",
                                 facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.1))
    ax1.text(x0 + 0.16, 0.28, lbl, color="#64748B", fontsize=7.5, fontweight="bold", ha="center", va="center")
    ax1.text(x0 + 0.16, 0.12, val, color=col, fontsize=11.5, fontweight="bold", ha="center", va="center")

ax2 = fig.add_axes([0.24, 0.10, 0.72, 0.40])
labels = ["Alemania", "España/MX", "Masculino", "CreditScore", "Edad", "Antigüedad", "Saldo", "Productos", "Salario"]
vals = X_sc[0]
colors = ["#4180AB" if v >= 0 else "#8AB3CF" for v in vals]
ax2.barh(labels, vals, color=colors, height=0.65)
ax2.axvline(0, color="#1E293B", linewidth=1.0)
ax2.set_xlim(-3.2, 3.2)
ax2.set_title("Vector Estandarizado (Z-Scores) del Cliente Nuevo enviado al SVM", fontsize=9.5, fontweight="bold", color="#1E293B", pad=6)
ax2.tick_params(axis="both", labelsize=8)
ax2.grid(axis="x", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.show()