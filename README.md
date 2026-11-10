# Máquinas de Soporte Vectorial (SVM) - Customer Churn Prediction
Proyecto Final - Ingeniería del Conocimiento | TecNM Campus Toluca
Docente: Ing. Rafael Hernández López | Entrega: 19 Nov 2026 | Defensa: 24 Nov 2026

## Asignación Modular del Equipo (7 Integrantes)
1. Apartado 1: src/preprocessing.py (Carga del CSV, EDA y SimpleImputer)
2. Apartado 2: src/preprocessing.py (OneHotEncoder, ColumnTransformer y matrices NumPy)
3. Apartado 3: src/preprocessing.py (train_test_split estratificado y StandardScaler)
4. Apartado 4: src/train.py (Ajuste de hiperparámetros C, gamma y SVC RBF)
5. Apartado 5: src/evaluate.py (Matriz de confusión, métricas y curva ROC-AUC)
6. Apartado 6: app/dashboard.py (Dashboard interactivo en Streamlit)
7. Apartado 7: models/*.joblib (Serialización con joblib, Git y README)

## Guía de Ejecución
1. pip install -r requirements.txt
2. python src/train.py
3. streamlit run app/dashboard.py
