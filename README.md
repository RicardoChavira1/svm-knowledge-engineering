# Proyecto Final: Predicción de Abandono de Clientes Bancarios (Churn) mediante Máquinas de Soporte Vectorial (SVM)

**Institución:** Tecnológico Nacional de México (TecNM) - Instituto Tecnológico de Toluca  
**Materia:** Ingeniería del Conocimiento (Grupo 219900 - Unidades 3 y 4)  
**Docente:** Ing. Rafael Hernández López  
**Fechas de Entrega:** Proyecto Técnico y Dashboard (19 de noviembre de 2026) | Presentación y Defensa (24 de noviembre de 2026)

## 1. Descripción General del Proyecto

Este proyecto implementa un ciclo completo de ciencia de datos e ingeniería del conocimiento (ETL, preprocesamiento estadístico, modelado predictivo, serialización de artefactos y visualización interactiva) aplicando el algoritmo de **Máquinas de Soporte Vectorial (Support Vector Machines - SVM)** para predecir la fuga o abandono de clientes (*Customer Churn*) en una institución bancaria.

## 2. Fuente, Procedencia y Diccionario de Datos (Fase 1)

### Procedencia del Conjunto de Datos
* **Nombre del Dataset:** *Bank Customer Churn Prediction* (Churn_Modelling.csv)
* **Fuente / Repositorio:** Kaggle (Repositorio público de conjuntos de datos estructurados para Machine Learning).
* **Volumen y Estructura:** 10,000 registros (filas) y 14 atributos (columnas) en formato CSV.
* **Objetivo del Negocio:** Identificar de manera anticipada qué clientes tienen alta probabilidad de cerrar su cuenta bancaria (Exited = 1) frente a aquellos que permanecerán en el banco (Exited = 0), permitiendo tomar acciones preventivas de retención.

### Diccionario de Datos Detallado

| Columna | Tipo | Escala | Descripción | Rol en SVM |
| :--- | :--- | :--- | :--- | :--- |
| RowNumber | Entero | Identificador | Número secuencial del registro (1 a 10,000) | Excluida |
| CustomerId | Entero | Identificador | ID único de la cuenta del cliente | Excluida |
| Surname | Texto | Nominal | Apellido del cliente | Excluida |
| CreditScore | Entero | Continua | Puntaje crediticio (350 a 850) | Predictora (X) |
| Geography | Texto | Nominal | País de residencia (France, Spain, Germany) | Predictora (X) |
| Gender | Texto | Nominal | Género registrado (Female, Male) | Predictora (X) |
| Age | Entero | Discreta | Edad del cliente en años cumplidos | Predictora (X) |
| Tenure | Entero | Discreta | Antigüedad en el banco en años | Predictora (X) |
| Balance | Flotante | Continua | Saldo monetario disponible en cuenta | Predictora (X) |
| NumOfProducts | Entero | Discreta | Productos bancarios contratados (1 a 4) | Predictora (X) |
| HasCrCard | Binario | Binaria | Posee tarjeta de crédito (1 = Sí, 0 = No) | Predictora (X) |
| IsActiveMember | Binario | Binaria | Es miembro activo (1 = Sí, 0 = No) | Predictora (X) |
| EstimatedSalary | Flotante | Continua | Salario anual estimado del cliente | Predictora (X) |
| Exited | Binario | Binaria | Abandonó el banco (1 = Fuga, 0 = Retiene) | Objetivo (y) |

## 3. Arquitectura del Proyecto y Flujo Técnico

    svm-knowledge-engineering/
    ├── Dashboard_Churn_SVM.pbix            # Tablero ejecutivo interactivo en Power BI (4 páginas)
    ├── tema_svm.json                       # Tema visual corporativo importado en Power BI
    ├── README.md                           # Documentación técnica, diccionario de datos y guía de uso
    ├── requirements.txt                    # Dependencias del entorno de Python
    ├── assets/                             # Plantillas HD (16:9) para el diseño del tablero Power BI
    ├── data/
    │   ├── raw/                            # Dataset original sin modificar (Churn_Modelling.csv)
    │   └── processed/                      # Datos limpios y tabla maestra para Power BI (churn_powerbi_dashboard.csv)
    ├── reports/                            # Reporte estadístico, matriz de confusión y curva ROC-AUC
    ├── src/
    │   ├── preprocessing.py                # Limpieza de duplicados, SimpleImputer, OneHotEncoder y StandardScaler
    │   ├── train.py                        # Optimización de kernels SVM, entrenamiento, serialización y exportación BI
    │   ├── evaluate.py                     # Cálculo de métricas, Matriz de Confusión y curva ROC-AUC
    │   └── generate_pbi_backgrounds.py     # Generador de plantillas visuales para Power BI
    ├── models/
    │   ├── svm_model.joblib                # Clasificador SVM entrenado y serializado
    │   ├── scaler.joblib                   # Objeto StandardScaler ajustado con datos de entrenamiento
    │   └── encoder.joblib                  # Objeto ColumnTransformer / OneHotEncoder ajustado
    └── app/
        └── dashboard.py                    # Simulador interactivo en Streamlit para inferencia en vivo

## 4. Asignación Modular del Equipo (7 Integrantes)

1. **Apartado 1 (Arquitectura y ETL):** src/preprocessing.py — Carga del dataset, análisis exploratorio (EDA), eliminación de duplicados e imputación de valores faltantes mediante SimpleImputer.
2. **Apartado 2 (Ingeniería de Características):** src/preprocessing.py — Transformación de variables categóricas nominales (OneHotEncoder, ColumnTransformer) y estructuración en matrices NumPy.
3. **Apartado 3 (Partición y Escalado):** src/preprocessing.py — División estratificada entrenamiento/prueba (80%/20%) con train_test_split y estandarización con StandardScaler previniendo fuga de datos (*Data Leakage*).
4. **Apartado 4 (Modelado SVM):** src/train.py — Configuración y entrenamiento de SVC con kernel RBF/lineal y optimización de hiperparámetros (C, gamma).
5. **Apartado 5 (Métricas y Evaluación):** src/evaluate.py — Matriz de confusión, reporte de clasificación (*Precision, Recall, F1-Score*) y curva ROC-AUC.
6. **Apartado 6 (Visualización y BI):** app/dashboard.py — Interfaz interactiva de inferencia en vivo y tablero de indicadores clave (KPIs).
7. **Apartado 7 (Serialización, Git y Documentación):** models/*.joblib y README.md — Persistencia de artefactos con joblib, control de versiones en GitHub, documentación técnica y presentación ejecutiva.

## 5. Guía de Instalación y Ejecución

1. Instalar las dependencias necesarias:

    pip install -r requirements.txt

2. Ejecutar el flujo de preprocesamiento, entrenamiento y serialización de artefactos:

    python src/train.py

3. Ejecutar la evaluación estadística y generación de gráficas del modelo:

    python src/evaluate.py

4. Iniciar la aplicación interactiva para pruebas en vivo:

    streamlit run app/dashboard.py