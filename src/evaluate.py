"""Apartado 5: evaluación de SVM para Customer Churn.

Clase 0: el cliente se queda.
Clase 1: el cliente abandona el banco.

X_test debe tener el mismo preprocesamiento usado al entrenar el modelo.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, output_dir="reports"):
    """Evalúa un modelo entrenado y guarda el reporte y dos gráficas."""
    y_test = np.asarray(y_test).ravel()

    if not np.array_equal(np.unique(y_test), [0, 1]):
        raise ValueError(
            "Los datos de prueba deben incluir las clases 0 y 1 "
            "para calcular la curva ROC y el AUC."
        )

    classes = np.asarray(model.classes_)
    if not np.array_equal(np.sort(classes), [0, 1]):
        raise ValueError(
            "El modelo debe estar entrenado con las clases 0 y 1."
        )

    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Entrena el SVC con probability=True.")

    # Predicciones y probabilidades de abandono.
    y_pred = model.predict(X_test)
    positive_index = list(classes).index(1)
    y_proba = model.predict_proba(X_test)[:, positive_index]

    # Filas: clases reales. Columnas: clases predichas.
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    vn, fp, fn, vp = matrix.ravel()

    names = ["Se queda (0)", "Se va (1)"]

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    text = (
        "EVALUACION DEL MODELO SVM\n"
        "Clase positiva: cliente que abandona el banco (1)\n\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        f"VN: {vn} | FP: {fp} | FN: {fn} | VP: {vp}\n\n"
        "Matriz de confusion (filas reales, columnas predichas):\n"
        f"{matrix}\n\n"
        f"Reporte de clasificacion:\n{report}"
    )
    print(text)

    # Crear la carpeta de resultados y guardar el reporte.
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "reporte_evaluacion.txt").write_text(
        text, encoding="utf-8"
    )

    # Gráfica de la matriz de confusión.
    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay(matrix, display_labels=names)
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Matriz de confusion - SVM")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    fig.tight_layout()
    fig.savefig(folder / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    # Curva ROC calculada con probabilidades, no con clases predichas.
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fpr, tpr, label=f"SVM (AUC = {auc:.4f})")
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Referencia al azar",
    )

    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title("Curva ROC - SVM")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(folder / "curva_roc.png", dpi=150)
    plt.close(fig)

    # Devolver resultados para que otros módulos puedan utilizarlos.
    return {
        "accuracy": float(accuracy),
        "precision": float(report_dict[names[1]]["precision"]),
        "recall": float(report_dict[names[1]]["recall"]),
        "f1": float(report_dict[names[1]]["f1-score"]),
        "roc_auc": float(auc),
        "confusion_matrix": matrix,
        "classification_report": report_dict,
    }
