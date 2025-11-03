"""
Pipeline Completo: Preparación de Datos + Modelado de Clasificación

Este script ejecuta el flujo completo del proyecto:
1. Carga datos originales de Google Play Store
2. Ejecuta pipeline de preparación de datos
3. Entrena modelos de clasificación (configurable por familia)
4. Evalúa en test y guarda resultados

Uso:
    python main.py

Familias disponibles:
- 'trees': RandomForest, ExtraTrees, GradientBoosting
- 'linear': LogisticRegression, SGDClassifier
- 'boosting': XGBoost, CatBoost

Autor: ML Team
Versión: 3.0 - Arquitectura modular con familias configurables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_data_preparation import GooglePlayDataPreparationPipeline
from pipeline_modelation import ClassificationTrainingPipeline

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
RANDOM_STATE = 42
DATA_PATH = '../../data/original/google-play-store/googleplaystore.csv'
OUTPUT_DIR = Path('./outputs')

# CONFIGURAR AQUÍ QUÉ FAMILIAS ENTRENAR
# Opciones: 'trees', 'linear', 'boosting'
# Se entrenarán Trees y Boosting con hiperparámetros optimizados anti-overfitting
FAMILIES_TO_TRAIN = ['trees', 'boosting']  # Cambiar según necesidad

# CONFIGURACIÓN PARA CLASIFICACIÓN BINARIA
N_BINS = 2  # Clasificación binaria: Low vs High
BINNING_STRATEGY = 'custom'  # Usar custom para definir el threshold
CUSTOM_BINS = [0.0, 4.0, 5.0]  # Low: [0.0-4.0], High: (4.0-5.0]
CUSTOM_LABELS = ['Low', 'High']

sns.set_style('whitegrid')

print("=" * 90)
print(" PIPELINE COMPLETO: PREPARACIÓN + MODELADO ".center(90, "="))
print("=" * 90)
print(f"\nFamilias a entrenar: {FAMILIES_TO_TRAIN}")
print(f"Tipo: Clasificación Binaria (Low ≤4.0 vs High >4.0)")
print("=" * 90)

# ==============================================================================
# PASO 1: PREPARACIÓN DE DATOS
# ==============================================================================
print("\n[PASO 1/3] Preparación de datos...")
print("-" * 90)

# Verificar si ya existen datos procesados
train_path = OUTPUT_DIR / 'train_processed.csv'
val_path = OUTPUT_DIR / 'val_processed.csv'
test_path = OUTPUT_DIR / 'test_processed.csv'

if train_path.exists() and val_path.exists() and test_path.exists():
    print("✓ Datos procesados encontrados. Cargando desde disco...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    print(f"   Train: {train.shape}")
    print(f"   Val:   {val.shape}")
    print(f"   Test:  {test.shape}")
else:
    print("⚠️  Datos procesados no encontrados. Ejecutando pipeline de preparación...")
    
    # Cargar datos originales
    print(f"\n  Cargando datos desde: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  ✓ Datos cargados: {df.shape}")
    
    # Ejecutar pipeline de preparación
    print("\n  Ejecutando GooglePlayDataPreparationPipeline...")
    data_pipeline = GooglePlayDataPreparationPipeline(
        test_size=0.30,
        val_size=0.50,
        category_threshold=70,
        mi_threshold=0.01,
        corr_threshold=0.8,
        vars_to_remove=['Installs_log'],
        reference_date='2025-10-02',
        random_state=RANDOM_STATE,
        verbose=True,
        plot=False
    )
    
    train, val, test = data_pipeline.fit_transform(df)
    
    # Guardar datasets procesados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"\n  ✓ Datos guardados en {OUTPUT_DIR}")

# ==============================================================================
# PASO 2: PREPARAR FEATURES Y TARGET
# ==============================================================================
print("\n[PASO 2/3] Preparando features y target...")
print("-" * 90)

columns_to_drop = ['Rating', 'App']
X_train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
y_train = train['Rating']

X_val = val.drop(columns=[col for col in columns_to_drop if col in val.columns])
y_val = val['Rating']

X_test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])
y_test = test['Rating']

print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Muestras: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ==============================================================================
# PASO 3: ENTRENAR MODELOS
# ==============================================================================
print("\n[PASO 3/3] Entrenando modelos...")
print("-" * 90)

start_time = time.time()

# Crear y entrenar pipeline de clasificación
classification_pipeline = ClassificationTrainingPipeline(
    n_bins=N_BINS,
    binning_strategy=BINNING_STRATEGY,
    custom_bins=CUSTOM_BINS,
    custom_labels=CUSTOM_LABELS,
    cv_folds=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=True,
    families=FAMILIES_TO_TRAIN
)

classification_pipeline.fit(X_train, y_train, X_val, y_val, X_test, y_test)
training_time = time.time() - start_time

print(f"\n✓ Entrenamiento completado en {training_time:.2f}s ({training_time/60:.2f} min)")

# ==============================================================================
# GUARDADO DE RESULTADOS
# ==============================================================================
print("\n" + "=" * 90)
print(" GUARDADO DE RESULTADOS ".center(90, "="))
print("=" * 90)

# Guardar resultados
results_dir = OUTPUT_DIR / 'classification'
results_dir.mkdir(parents=True, exist_ok=True)

classification_pipeline.save_results(output_dir=str(results_dir))

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
print("\n" + "=" * 90)
print(" ✅ PIPELINE COMPLETADO EXITOSAMENTE ".center(90, "="))
print("=" * 90)

print(f"\n📁 Resultados guardados en: {results_dir}")
print(f"⏱️  Tiempo total: {training_time:.2f}s ({training_time/60:.2f} min)")

print("\n" + "=" * 90)

