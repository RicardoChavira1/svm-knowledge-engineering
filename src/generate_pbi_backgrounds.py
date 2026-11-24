"""
Generador automático de 7 plantillas ejecutivas (16:9 HD) para Power BI.
Alineado a la defensa del 24 de noviembre de 2026 y la paleta oficial del proyecto SVM.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

OUT_DIR = Path("assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG_LIGHT = "#E4EBF0"
NAVY = "#1E293B"
NAVY_DARK = "#0F172A"
BLUE = "#4180AB"
CYAN = "#38BDF8"
WHITE = "#FFFFFF"
BORDER = "#CBD5E1"
TEXT_DARK = "#1E293B"
TEXT_MUTED = "#475569"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#16A34A"


def new_canvas(bg_color=BG_LIGHT):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=bg_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor="none", zorder=0))
    return fig, ax


def add_card(ax, x, y, w, h, facecolor=WHITE, edgecolor=BORDER, radius=0.015, lw=1.2, zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.003, y - 0.005),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor="#94A3B8",
            edgecolor="none",
            alpha=0.22,
            zorder=zorder - 0.5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            zorder=zorder,
        )
    )


def add_sidebar_and_header(ax, title, subtitle):
    # Barra lateral oscura dimensionada para 7 botones verticales
    ax.add_patch(Rectangle((0, 0), 0.16, 1, facecolor=NAVY, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((0.158, 0), 0.003, 1, facecolor=BLUE, edgecolor="none", zorder=3))

    ax.text(0.08, 0.945, "SVM ANALYTICS", color=CYAN, fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(0.08, 0.910, "CHURN BANCARIO", color="#CBD5E1", fontsize=8.2, ha="center", va="center", zorder=4)
    ax.plot([0.02, 0.14], [0.880, 0.880], color="#334155", lw=1.2, zorder=4)
    ax.text(0.08, 0.855, "MENÚ PRINCIPAL", color="#94A3B8", fontsize=7.8, fontweight="bold", ha="center", va="center", zorder=4)

    # Pie compacto para dejar máximo espacio al navegador (y=0.095 a y=0.835)
    ax.plot([0.02, 0.14], [0.088, 0.088], color="#334155", lw=1.0, zorder=4)
    ax.text(0.08, 0.058, "TecNM · Campus Toluca", color="#CBD5E1", fontsize=7.8, ha="center", va="center", zorder=4)
    ax.text(0.08, 0.033, "Ing. del Conocimiento", color="#94A3B8", fontsize=7.3, ha="center", va="center", zorder=4)

    # Encabezado superior derecho
    add_card(ax, 0.175, 0.895, 0.81, 0.085, facecolor=WHITE, edgecolor=BORDER, radius=0.012, zorder=2)
    ax.add_patch(Rectangle((0.175, 0.895), 0.006, 0.085, facecolor=BLUE, edgecolor="none", zorder=3))
    ax.text(0.195, 0.952, title, color=TEXT_DARK, fontsize=16.5, fontweight="bold", ha="left", va="center", zorder=4)
    ax.text(0.195, 0.916, subtitle, color=TEXT_MUTED, fontsize=9.6, ha="left", va="center", zorder=4)


def add_filter_bar(ax, label="FILTROS\nDINÁMICOS"):
    add_card(ax, 0.175, 0.795, 0.81, 0.082, facecolor="#F8FAFC", edgecolor=BORDER, radius=0.012, zorder=2)
    ax.text(0.19, 0.836, label, color=BLUE, fontsize=8.3, fontweight="bold", ha="left", va="center", zorder=3)


def build_portada():
    fig, ax = new_canvas("#F8FAFC")

    ax.add_patch(Rectangle((0.49, 0), 0.51, 1, facecolor=NAVY_DARK, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0.49, 0), 0.005, 1, facecolor=CYAN, edgecolor="none", zorder=3))

    for gx in np.linspace(0.52, 0.98, 12):
        ax.plot([gx, gx], [0.05, 0.95], color="#1E293B", lw=0.8, zorder=2)
    for gy in np.linspace(0.08, 0.92, 10):
        ax.plot([0.51, 0.98], [gy, gy], color="#1E293B", lw=0.8, zorder=2)

    rng = np.random.default_rng(42)
    t = np.linspace(0.54, 0.96, 200)
    y_center = 0.56 + 0.14 * np.sin((t - 0.54) * 9.5)
    ax.fill_between(t, y_center - 0.08, y_center + 0.08, color=CYAN, alpha=0.10, zorder=3)
    ax.plot(t, y_center, color=CYAN, lw=2.8, zorder=4)
    ax.plot(t, y_center + 0.08, color=CYAN, lw=1.3, ls="--", alpha=0.7, zorder=4)
    ax.plot(t, y_center - 0.08, color=CYAN, lw=1.3, ls="--", alpha=0.7, zorder=4)

    x_safe = rng.uniform(0.54, 0.94, 48)
    y_safe = 0.56 + 0.14 * np.sin((x_safe - 0.54) * 9.5) - rng.uniform(0.11, 0.27, 48)
    x_risk = rng.uniform(0.54, 0.94, 34)
    y_risk = 0.56 + 0.14 * np.sin((x_risk - 0.54) * 9.5) + rng.uniform(0.11, 0.26, 34)
    ax.scatter(x_safe, y_safe, s=55, color=BLUE, edgecolors=CYAN, linewidths=0.8, alpha=0.9, zorder=5)
    ax.scatter(x_risk, y_risk, s=55, color=RED, edgecolors="#FCA5A5", linewidths=0.8, alpha=0.9, zorder=5)

    ax.text(0.745, 0.91, "HIPERPLANO NO LINEAL · KERNEL RBF", color=CYAN, fontsize=11, fontweight="bold", ha="center", zorder=6)
    ax.text(0.745, 0.875, "Separación de cartera bancaria en espacio de alta dimensión", color="#94A3B8", fontsize=9.5, ha="center", zorder=6)

    stats = [
        ("10,000", "Clientes Analizados"),
        ("85.40%", "Exactitud SVM (RBF)"),
        ("0.8114", "Área ROC-AUC"),
    ]
    for i, (val, lbl) in enumerate(stats):
        bx = 0.525 + i * 0.152
        add_card(ax, bx, 0.09, 0.138, 0.135, facecolor="#1E293B", edgecolor="#334155", radius=0.014, lw=1.2, zorder=5)
        ax.text(bx + 0.069, 0.175, val, color=CYAN, fontsize=16, fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(bx + 0.069, 0.125, lbl, color="#E2E8F0", fontsize=8.5, ha="center", va="center", zorder=6)

    add_card(ax, 0.05, 0.855, 0.34, 0.048, facecolor="#E0F2FE", edgecolor="#7DD3FC", radius=0.015, lw=1.0, zorder=3)
    ax.text(0.22, 0.879, "TECNM CAMPUS TOLUCA  •  ING. DEL CONOCIMIENTO", color=BLUE, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=4)

    ax.text(0.05, 0.77, "Predicción de Fuga Bancaria:\nModelo SVM (Kernel RBF)", color=NAVY_DARK, fontsize=25, fontweight="bold", ha="left", va="center", linespacing=1.2, zorder=4)
    ax.text(
        0.05,
        0.645,
        "Informe ejecutivo y diagnóstico predictivo sobre 10,000 clientes\n"
        "bancarios para identificar perfiles en riesgo de abandono y\n"
        "evaluar clientes nuevos en tiempo real (MXN).",
        color=TEXT_MUTED,
        fontsize=11,
        ha="left",
        va="center",
        linespacing=1.45,
        zorder=4,
    )

    add_card(ax, 0.05, 0.255, 0.40, 0.30, facecolor=WHITE, edgecolor=BORDER, radius=0.014, zorder=3)
    ax.text(0.07, 0.525, "DOCENTE: Ing. Rafael Hernández López  |  Grupo 219900", color=BLUE, fontsize=9.2, fontweight="bold", ha="left", va="center", zorder=4)
    ax.plot([0.07, 0.43], [0.50, 0.50], color=BORDER, lw=1.0, zorder=4)
    ax.text(0.07, 0.472, "INTEGRANTES DEL EQUIPO:", color=TEXT_DARK, fontsize=9, fontweight="bold", ha="left", va="center", zorder=4)

    integrantes = [
        "• Ricardo Lugo Chavira",
        "• Deni Samanta Filomeno Camacho",
        "• Vania Guadalupe González Segura",
        "• Rafael David Mota López",
        "• Eric Díaz Pérez",
        "• Alan Ramses González Ruíz",
        "• José Emilio Esquivel Narváez",
    ]
    ax.text(0.075, 0.362, "\n".join(integrantes), color=TEXT_MUTED, fontsize=9.3, ha="left", va="center", linespacing=1.42, zorder=4)

    ax.text(0.05, 0.06, "Presentación Ejecutiva y Defensa: 24 de noviembre de 2026", color="#64748B", fontsize=8.8, ha="left", va="center", zorder=4)

    fig.savefig(OUT_DIR / "bg_portada.png", dpi=120)
    plt.close(fig)


def build_clientes():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Análisis de Clientes y Perfiles de Riesgo de Fuga",
        "Indicadores clave (KPIs), segmentación demográfica y Top clientes expuestos en MXN",
    )
    add_filter_bar(ax)
    fig.savefig(OUT_DIR / "bg_clientes.png", dpi=120)
    # También sobreescribe bg_dashboard.png para mantener compatibilidad inmediata
    fig.savefig(OUT_DIR / "bg_dashboard.png", dpi=120)
    plt.close(fig)


def build_geografico():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Análisis Geográfico y Comportamiento Regional",
        "Comparativa de abandono real vs. predicción SVM y capital en riesgo por mercado (Francia, Alemania, España)",
    )
    add_filter_bar(ax, label="FILTROS\nREGIONALES")
    fig.savefig(OUT_DIR / "bg_geografico.png", dpi=120)
    plt.close(fig)


def build_productos():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Análisis de Productos Bancarios y Perfil Financiero",
        "Impacto de la tenencia de productos (70.15% riesgo en monoproducto), buró de crédito y antigüedad",
    )
    add_filter_bar(ax, label="FILTROS\nFINANCIEROS")
    fig.savefig(OUT_DIR / "bg_productos.png", dpi=120)
    plt.close(fig)


def build_diagnostico():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Diagnóstico Estadístico y Optimización del Modelo SVM",
        "Comparativa de hiperparámetros por kernel (Conjunto de prueba estratificado: 2,000 clientes)",
    )

    add_card(ax, 0.175, 0.675, 0.81, 0.20, facecolor=WHITE, edgecolor=BORDER, radius=0.014, zorder=2)
    ax.text(0.195, 0.845, "COMPARATIVA DE OPTIMIZACIÓN DE KERNELS (C = 1.0, gamma = 'scale')", color=BLUE, fontsize=10.5, fontweight="bold", ha="left", va="center", zorder=3)

    cols_x = [0.20, 0.34, 0.47, 0.60, 0.72]
    headers = ["Kernel Evaluado", "Exactitud (Acc)", "F1-Score (Fuga)", "Área ROC-AUC", "Dictamen Técnico"]
    ax.add_patch(Rectangle((0.195, 0.795), 0.77, 0.032, facecolor="#F1F5F9", edgecolor="none", zorder=3))
    for cx, h in zip(cols_x, headers):
        ax.text(cx, 0.811, h, color=TEXT_DARK, fontsize=9, fontweight="bold", ha="left", va="center", zorder=4)

    rows = [
        ("LINEAR", "79.65%", "0.0000", "0.6501", "Descartado: no detecta clase fuga", TEXT_MUTED, False),
        ("POLY", "84.75%", "0.4804", "0.8291", "Alternativa no lineal competitiva", TEXT_MUTED, False),
        ("RBF (ELEGIDO)", "85.40%", "0.5197", "0.8114", "Óptimo equilibrio Accuracy + F1", NAVY_DARK, True),
    ]
    for idx, (k, acc, f1, auc, note, col, highlight) in enumerate(rows):
        ry = 0.768 - idx * 0.036
        if highlight:
            ax.add_patch(Rectangle((0.195, ry - 0.016), 0.77, 0.032, facecolor="#E0F2FE", edgecolor="none", zorder=3))
        weight = "bold" if highlight else "normal"
        for cx, val in zip(cols_x, [k, acc, f1, auc, note]):
            ax.text(cx, ry, val, color=BLUE if (highlight and cx == 0.20) else col, fontsize=9.2, fontweight=weight, ha="left", va="center", zorder=4)

    fig.savefig(OUT_DIR / "bg_diagnostico.png", dpi=120)
    plt.close(fig)


def build_nuevos():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Predicción en Vivo: Clientes Completamente Nuevos (Fuera del Dataset)",
        "Inferencia directa con modelos .joblib: simulador interactivo en tiempo real + registro de casos nuevos",
    )

    # Zona izquierda: Simulador en vivo con parámetros + Visual Python
    add_card(ax, 0.175, 0.035, 0.445, 0.840, facecolor="#F8FAFC", edgecolor=BORDER, radius=0.014, zorder=2)
    ax.add_patch(Rectangle((0.175, 0.863), 0.445, 0.012, facecolor=BLUE, edgecolor="none", zorder=3))
    ax.text(0.192, 0.835, "OPCIÓN A · SIMULADOR SVM EN VIVO (PYTHON EN POWER BI)", color=BLUE, fontsize=9.5, fontweight="bold", ha="left", va="center", zorder=4)

    # Zona derecha: Registro / Lote de nuevos clientes evaluados desde CLI
    add_card(ax, 0.635, 0.035, 0.350, 0.840, facecolor="#F8FAFC", edgecolor=BORDER, radius=0.014, zorder=2)
    ax.add_patch(Rectangle((0.635, 0.863), 0.350, 0.012, facecolor=NAVY, edgecolor="none", zorder=3))
    ax.text(0.652, 0.835, "OPCIÓN B · REGISTRO DE NUEVOS CASOS", color=NAVY, fontsize=9.5, fontweight="bold", ha="left", va="center", zorder=4)

    fig.savefig(OUT_DIR / "bg_nuevos.png", dpi=120)
    plt.close(fig)


def build_conclusiones():
    fig, ax = new_canvas(BG_LIGHT)
    add_sidebar_and_header(
        ax,
        "Conclusiones Ejecutivas y Estrategia de Retención",
        "Hallazgos accionables derivados del clasificador SVM (Kernel RBF) para la toma de decisiones",
    )

    cards = [
        (
            0.175,
            RED,
            "01  •  EXPOSICIÓN FINANCIERA",
            "Capital Concentrado en Riesgo",
            "$1.75 mil M MXN",
            "El modelo identificó 985 clientes en\n"
            "riesgo crítico de abandono sobre una\n"
            "cartera de 10,000 cuentas.\n\n"
            "La probabilidad media de fuga global es\n"
            "de 20.48%, lo que evidencia la necesidad\n"
            "de intervención focalizada.",
        ),
        (
            0.452,
            AMBER,
            "02  •  PERFIL DE ALTA VULNERABILIDAD",
            "Monoproducto, Edad y Región",
            "70.15% del Riesgo",
            "$1.23 mil M MXN del capital expuesto\n"
            "pertenece a clientes con un solo\n"
            "producto bancario contratado.\n\n"
            "La mayor densidad de riesgo crítico se\n"
            "concentra en el segmento Maduro\n"
            "(46-60 años) y en Alemania.",
        ),
        (
            0.729,
            BLUE,
            "03  •  DESEMPEÑO DEL MODELO",
            "Superioridad del Kernel RBF",
            "85.40% Exactitud",
            "El kernel RBF superó al modelo lineal\n"
            "(79.65%) y polinomial (84.75%), logrando\n"
            "un ROC-AUC de 0.8114.\n\n"
            "Alcanza 78.61% de precisión al alertar\n"
            "fugas reales y evalúa clientes nuevos\n"
            "en tiempo real con los artefactos .joblib.",
        ),
    ]

    for cx, accent, tag, title, big_num, body in cards:
        add_card(ax, cx, 0.34, 0.256, 0.53, facecolor=WHITE, edgecolor=BORDER, radius=0.015, zorder=2)
        ax.add_patch(Rectangle((cx, 0.858), 0.256, 0.012, facecolor=accent, edgecolor="none", zorder=3))
        ax.text(cx + 0.018, 0.822, tag, color=accent, fontsize=8.5, fontweight="bold", ha="left", va="center", zorder=4)
        ax.text(cx + 0.018, 0.778, title, color=TEXT_DARK, fontsize=13, fontweight="bold", ha="left", va="center", zorder=4)
        ax.text(cx + 0.018, 0.705, big_num, color=accent, fontsize=20, fontweight="bold", ha="left", va="center", zorder=4)
        ax.plot([cx + 0.018, cx + 0.238], [0.655, 0.655], color=BORDER, lw=1.0, zorder=4)
        ax.text(cx + 0.018, 0.505, body, color=TEXT_MUTED, fontsize=9.8, ha="left", va="center", linespacing=1.48, zorder=4)

    add_card(ax, 0.175, 0.045, 0.81, 0.265, facecolor=NAVY, edgecolor=NAVY, radius=0.015, zorder=2)
    ax.add_patch(Rectangle((0.175, 0.045), 0.008, 0.265, facecolor=GREEN, edgecolor="none", zorder=3))
    ax.text(0.20, 0.27, "PLAN DE ACCIÓN RECOMENDADO PARA EL BANCO", color=CYAN, fontsize=11.5, fontweight="bold", ha="left", va="center", zorder=4)

    acciones = [
        "1. Campaña de Venta Cruzada (Cross-Selling): Ofrecer un segundo producto (tarjeta sin anualidad o inversión) a clientes monoproducto con saldo alto para reducir su tasa de abandono.",
        "2. Atención Prioritaria a Segmento Maduro (46-60 años) y Alemania: Asignar ejecutivos de retención a clientes cuya probabilidad SVM supere el 65% (Riesgo Crítico).",
        "3. Evaluación Preventiva de Nuevos Perfiles: Utilizar el módulo de predicción en vivo integrado en Power BI para puntuar clientes nuevos y anticipar fugas antes de la cancelación.",
    ]
    ax.text(0.20, 0.15, "\n\n".join(acciones), color="#E2E8F0", fontsize=9.8, ha="left", va="center", linespacing=1.35, zorder=4)

    fig.savefig(OUT_DIR / "bg_conclusiones.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    build_portada()
    build_clientes()
    build_geografico()
    build_productos()
    build_diagnostico()
    build_nuevos()
    build_conclusiones()
    print("[+] 7 plantillas HD generadas correctamente en 'assets/'")