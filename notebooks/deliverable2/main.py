"""
Pipeline Completo: Preparación de Datos + Clasificación Binaria (Low vs High)

Este script ejecuta el flujo completo del proyecto:
1. Carga datos originales de Google Play Store
2. Ejecuta pipeline de preparación de datos
3. Entrena Voting Ensemble (RandomForest + XGBoost + CatBoost)
4. Calibra threshold para optimizar F1-macro
5. Evalúa en test y guarda resultados

Autor: ML Team
Versión: 2.0 - Clasificación Binaria con Voting Ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_data_preparation import GooglePlayDataPreparationPipeline

# Configuración global
RANDOM_STATE = 42
DATA_PATH = '../../data/original/google-play-store/googleplaystore.csv'
OUTPUT_DIR = Path('./outputs')
sns.set_style('whitegrid')

print("=" * 90)
print(" PIPELINE COMPLETO: PREPARACIÓN + CLASIFICACIÓN BINARIA ".center(90, "="))
print("=" * 90)
print(f"\nObjetivo: Clasificar Rating en Low (≤4.0) vs High (>4.0)")
print(f"Modelo: Voting Ensemble (RF + XGBoost + CatBoost) con threshold calibrado")
print("=" * 90)

# ==============================================================================
# PASO 1: PREPARACIÓN DE DATOS
# ==============================================================================
print("\n[PASO 1/4] Preparación de datos...")
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
print("\n[PASO 2/4] Preparando features y target...")
print("-" * 90)

# Separar features y target (clasificación binaria)
columns_to_drop = ['Rating', 'App']
X_train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])
y_train = (train['Rating'] > 4.0).astype(int)  # 0=Low (≤4.0), 1=High (>4.0)

X_val = val.drop(columns=[col for col in columns_to_drop if col in val.columns])
y_val = (val['Rating'] > 4.0).astype(int)

X_test = test.drop(columns=[col for col in columns_to_drop if col in test.columns])
y_test = (test['Rating'] > 4.0).astype(int)

print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Muestras: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"\nDistribución de clases:")
print(f"  Train - Low: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%), High: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)")
print(f"  Val   - Low: {(y_val==0).sum()} ({(y_val==0).mean()*100:.1f}%), High: {(y_val==1).sum()} ({(y_val==1).mean()*100:.1f}%)")
print(f"  Test  - Low: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%), High: {(y_test==1).sum()} ({(y_test==1).mean()*100:.1f}%)")

# Escalar datos
print("\n✓ Escalando features...")
scaler_standard = StandardScaler()
X_train_scaled = scaler_standard.fit_transform(X_train)
X_val_scaled = scaler_standard.transform(X_val)
X_test_scaled = scaler_standard.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_catboost = scaler_minmax.fit_transform(X_train)
X_val_catboost = scaler_minmax.transform(X_val)
X_test_catboost = scaler_minmax.transform(X_test)

# ==============================================================================
# PASO 3: ENTRENAR VOTING ENSEMBLE
# ==============================================================================
print("\n[PASO 3/4] Entrenando Voting Ensemble...")
print("-" * 90)

results_individual = []

# RandomForest
print("\n🌲 Entrenando RandomForest...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time

proba_val_rf = rf_model.predict_proba(X_val_scaled)
proba_test_rf = rf_model.predict_proba(X_test_scaled)
y_pred_val_rf = rf_model.predict(X_val_scaled)
y_pred_test_rf = rf_model.predict(X_test_scaled)

p_val, r_val, f_val, _ = precision_recall_fscore_support(y_val, y_pred_val_rf, labels=[0, 1], zero_division=0)
p_test, r_test, f_test, _ = precision_recall_fscore_support(y_test, y_pred_test_rf, labels=[0, 1], zero_division=0)

results_individual.append({
    'Model': 'RandomForest',
    'Time_s': rf_time,
    'Precision_Low_val': p_val[0],
    'Recall_Low_val': r_val[0],
    'F1_macro_val': (f_val[0] + f_val[1]) / 2,
    'Precision_Low_test': p_test[0],
    'Recall_Low_test': r_test[0],
    'F1_macro_test': (f_test[0] + f_test[1]) / 2,
    'Accuracy_test': accuracy_score(y_test, y_pred_test_rf)
})

print(f"✓ Completado en {rf_time:.2f}s")
print(f"  Val  - Precision Low: {p_val[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_val']:.4f}")
print(f"  Test - Precision Low: {p_test[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_test']:.4f}")

# XGBoost
print("\n⚡ Entrenando XGBoost...")
start_time = time.time()
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=1.0,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_time = time.time() - start_time

proba_val_xgb = xgb_model.predict_proba(X_val_scaled)
proba_test_xgb = xgb_model.predict_proba(X_test_scaled)
y_pred_val_xgb = xgb_model.predict(X_val_scaled)
y_pred_test_xgb = xgb_model.predict(X_test_scaled)

p_val, r_val, f_val, _ = precision_recall_fscore_support(y_val, y_pred_val_xgb, labels=[0, 1], zero_division=0)
p_test, r_test, f_test, _ = precision_recall_fscore_support(y_test, y_pred_test_xgb, labels=[0, 1], zero_division=0)

results_individual.append({
    'Model': 'XGBoost',
    'Time_s': xgb_time,
    'Precision_Low_val': p_val[0],
    'Recall_Low_val': r_val[0],
    'F1_macro_val': (f_val[0] + f_val[1]) / 2,
    'Precision_Low_test': p_test[0],
    'Recall_Low_test': r_test[0],
    'F1_macro_test': (f_test[0] + f_test[1]) / 2,
    'Accuracy_test': accuracy_score(y_test, y_pred_test_xgb)
})

print(f"✓ Completado en {xgb_time:.2f}s")
print(f"  Val  - Precision Low: {p_val[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_val']:.4f}")
print(f"  Test - Precision Low: {p_test[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_test']:.4f}")

# CatBoost
print("\n🐱 Entrenando CatBoost...")
start_time = time.time()
catboost_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=1,
    auto_class_weights='SqrtBalanced',
    random_state=RANDOM_STATE,
    verbose=0,
    thread_count=-1
)
catboost_model.fit(X_train_catboost, y_train)
catboost_time = time.time() - start_time

proba_val_cat = catboost_model.predict_proba(X_val_catboost)
proba_test_cat = catboost_model.predict_proba(X_test_catboost)
y_pred_val_cat = catboost_model.predict(X_val_catboost)
y_pred_test_cat = catboost_model.predict(X_test_catboost)

p_val, r_val, f_val, _ = precision_recall_fscore_support(y_val, y_pred_val_cat, labels=[0, 1], zero_division=0)
p_test, r_test, f_test, _ = precision_recall_fscore_support(y_test, y_pred_test_cat, labels=[0, 1], zero_division=0)

results_individual.append({
    'Model': 'CatBoost',
    'Time_s': catboost_time,
    'Precision_Low_val': p_val[0],
    'Recall_Low_val': r_val[0],
    'F1_macro_val': (f_val[0] + f_val[1]) / 2,
    'Precision_Low_test': p_test[0],
    'Recall_Low_test': r_test[0],
    'F1_macro_test': (f_test[0] + f_test[1]) / 2,
    'Accuracy_test': accuracy_score(y_test, y_pred_test_cat)
})

print(f"✓ Completado en {catboost_time:.2f}s")
print(f"  Val  - Precision Low: {p_val[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_val']:.4f}")
print(f"  Test - Precision Low: {p_test[0]:.4f}, F1-macro: {results_individual[-1]['F1_macro_test']:.4f}")

# Voting Ensemble (Soft)
print("\n🗳️  Creando Voting Ensemble (Soft)...")
proba_val_avg = (proba_val_rf + proba_val_xgb + proba_val_cat) / 3
proba_test_avg = (proba_test_rf + proba_test_xgb + proba_test_cat) / 3

y_proba_val_soft = proba_val_avg[:, 1]
y_proba_test_soft = proba_test_avg[:, 1]

y_pred_val_soft = np.argmax(proba_val_avg, axis=1)
y_pred_test_soft = np.argmax(proba_test_avg, axis=1)

p_val, r_val, f_val, _ = precision_recall_fscore_support(y_val, y_pred_val_soft, labels=[0, 1], zero_division=0)
p_test, r_test, f_test, _ = precision_recall_fscore_support(y_test, y_pred_test_soft, labels=[0, 1], zero_division=0)

results_voting_soft = {
    'Model': 'Voting (Soft)',
    'Time_s': 0.0,
    'Precision_Low_val': p_val[0],
    'Recall_Low_val': r_val[0],
    'F1_macro_val': (f_val[0] + f_val[1]) / 2,
    'Precision_Low_test': p_test[0],
    'Recall_Low_test': r_test[0],
    'F1_macro_test': (f_test[0] + f_test[1]) / 2,
    'Accuracy_test': accuracy_score(y_test, y_pred_test_soft)
}

print(f"✓ Soft Voting creado")
print(f"  Val  - Precision Low: {p_val[0]:.4f}, F1-macro: {results_voting_soft['F1_macro_val']:.4f}")
print(f"  Test - Precision Low: {p_test[0]:.4f}, F1-macro: {results_voting_soft['F1_macro_test']:.4f}")

# Threshold Calibration
print("\n🎯 Calibrando threshold para optimizar F1-macro...")
thresholds = np.arange(0.3, 0.8, 0.01)
best_f1_macro = 0
best_threshold = 0.5

for thresh in thresholds:
    y_pred_temp = (y_proba_val_soft >= thresh).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_val, y_pred_temp, labels=[0, 1], zero_division=0)
    f1_macro = (f[0] + f[1]) / 2
    
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_threshold = thresh

print(f"✓ Mejor threshold: {best_threshold:.3f} (F1-macro val: {best_f1_macro:.4f})")

# Aplicar threshold calibrado
y_pred_test_calibrated = (y_proba_test_soft >= best_threshold).astype(int)

p_test_cal, r_test_cal, f_test_cal, _ = precision_recall_fscore_support(
    y_test, y_pred_test_calibrated, labels=[0, 1], zero_division=0
)

results_voting_calibrated = {
    'Model': 'Voting (Soft + Calibrated)',
    'Time_s': 0.0,
    'Precision_Low_val': results_voting_soft['Precision_Low_val'],
    'Recall_Low_val': results_voting_soft['Recall_Low_val'],
    'F1_macro_val': best_f1_macro,
    'Precision_Low_test': p_test_cal[0],
    'Recall_Low_test': r_test_cal[0],
    'F1_macro_test': (f_test_cal[0] + f_test_cal[1]) / 2,
    'Accuracy_test': accuracy_score(y_test, y_pred_test_calibrated)
}

print(f"\n✓ Con threshold calibrado:")
print(f"  Test - Precision Low: {p_test_cal[0]:.4f}, Recall Low: {r_test_cal[0]:.4f}")
print(f"  Test - F1-macro: {results_voting_calibrated['F1_macro_test']:.4f}, Accuracy: {results_voting_calibrated['Accuracy_test']:.4f}")

# ==============================================================================
# PASO 4: EVALUACIÓN Y RESULTADOS
# ==============================================================================
print("\n[PASO 4/4] Evaluación final y guardado de resultados...")
print("-" * 90)

# Consolidar resultados
all_results = results_individual + [results_voting_soft, results_voting_calibrated]
results_df = pd.DataFrame(all_results)

# Identificar mejor modelo
best_model_row = results_df.loc[results_df['F1_macro_test'].idxmax()]

print("\n" + "=" * 90)
print(" MEJOR MODELO ".center(90, "="))
print("=" * 90)
print(f"\n🏆 {best_model_row['Model']}")
print(f"   Precision Low: {best_model_row['Precision_Low_test']:.4f}")
print(f"   Recall Low:    {best_model_row['Recall_Low_test']:.4f}")
print(f"   F1-macro:      {best_model_row['F1_macro_test']:.4f}")
print(f"   Accuracy:      {best_model_row['Accuracy_test']:.4f}")

# Mejora vs mejor modelo individual
best_individual = results_df[results_df['Model'].isin(['RandomForest', 'XGBoost', 'CatBoost'])]['F1_macro_test'].max()
improvement = best_model_row['F1_macro_test'] - best_individual

print(f"\n📈 Mejora sobre mejor modelo individual:")
print(f"   F1-macro: {best_individual:.4f} → {best_model_row['F1_macro_test']:.4f} ({improvement*100:+.2f}%)")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test_calibrated)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(f"\n📊 Matriz de Confusión (Normalizada):")
print(f"              Predicho Low  Predicho High")
print(f"  Real Low    {cm_normalized[0,0]:.2f}          {cm_normalized[0,1]:.2f}")
print(f"  Real High   {cm_normalized[1,0]:.2f}          {cm_normalized[1,1]:.2f}")

# Reporte de clasificación
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test_calibrated, target_names=['Low', 'High']))

# ==============================================================================
# GUARDAR RESULTADOS
# ==============================================================================
print("\n" + "=" * 90)
print(" GUARDANDO RESULTADOS ".center(90, "="))
print("=" * 90)

# Crear directorio de clasificación
classification_dir = OUTPUT_DIR / 'classification'
classification_dir.mkdir(parents=True, exist_ok=True)

# 1. Comparación de modelos
results_df.to_csv(classification_dir / 'model_comparison.csv', index=False)
print(f"✓ {classification_dir / 'model_comparison.csv'}")

# 2. Predicciones de test
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_test_calibrated,
    'proba_Low': 1 - y_proba_test_soft,
    'proba_High': y_proba_test_soft,
    'correct': y_test == y_pred_test_calibrated
})
predictions_df.to_csv(classification_dir / 'test_predictions.csv', index=False)
print(f"✓ {classification_dir / 'test_predictions.csv'}")

# 3. Métricas finales
final_metrics = pd.DataFrame({
    'Dataset': ['Val', 'Test'],
    'Accuracy': [
        accuracy_score(y_val, (y_proba_val_soft >= best_threshold).astype(int)),
        results_voting_calibrated['Accuracy_test']
    ],
    'Precision_Low': [
        results_voting_calibrated['Precision_Low_val'],
        results_voting_calibrated['Precision_Low_test']
    ],
    'Recall_Low': [
        results_voting_calibrated['Recall_Low_val'],
        results_voting_calibrated['Recall_Low_test']
    ],
    'F1_macro': [
        results_voting_calibrated['F1_macro_val'],
        results_voting_calibrated['F1_macro_test']
    ]
})
final_metrics.to_csv(classification_dir / 'final_metrics.csv', index=False)
print(f"✓ {classification_dir / 'final_metrics.csv'}")

# 4. Matriz de confusión
cm_df = pd.DataFrame(cm, index=['Real_Low', 'Real_High'], columns=['Pred_Low', 'Pred_High'])
cm_df.to_csv(classification_dir / 'confusion_matrix.csv')
print(f"✓ {classification_dir / 'confusion_matrix.csv'}")

# 5. Resumen en texto
with open(classification_dir / 'training_summary.txt', 'w') as f:
    f.write("=" * 90 + "\n")
    f.write(" RESUMEN DE ENTRENAMIENTO - CLASIFICACIÓN BINARIA ".center(90, "=") + "\n")
    f.write("=" * 90 + "\n\n")
    
    f.write("CONFIGURACIÓN:\n")
    f.write("-" * 90 + "\n")
    f.write(f"  Problema: Clasificación Binaria (Low ≤4.0 vs High >4.0)\n")
    f.write(f"  Modelo: Voting Ensemble (RF + XGBoost + CatBoost)\n")
    f.write(f"  Threshold: {best_threshold:.3f} (optimizado para F1-macro)\n")
    f.write(f"  Features: {X_train.shape[1]}\n")
    f.write(f"  Train samples: {len(X_train)}\n")
    f.write(f"  Val samples: {len(X_val)}\n")
    f.write(f"  Test samples: {len(X_test)}\n\n")
    
    f.write("RESULTADOS:\n")
    f.write("-" * 90 + "\n")
    f.write(f"  Mejor modelo: {best_model_row['Model']}\n")
    f.write(f"  F1-macro (test): {best_model_row['F1_macro_test']:.4f}\n")
    f.write(f"  Accuracy (test): {best_model_row['Accuracy_test']:.4f}\n")
    f.write(f"  Precision Low: {best_model_row['Precision_Low_test']:.4f}\n")
    f.write(f"  Recall Low: {best_model_row['Recall_Low_test']:.4f}\n\n")
    
    f.write("COMPARACIÓN DE MODELOS:\n")
    f.write("-" * 90 + "\n")
    f.write(results_df[['Model', 'F1_macro_test', 'Precision_Low_test', 'Accuracy_test']].to_string(index=False))
    f.write("\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 90 + "\n")
    f.write(classification_report(y_test, y_pred_test_calibrated, target_names=['Low', 'High']))

print(f"✓ {classification_dir / 'training_summary.txt'}")

# Visualizaciones
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title(f'Confusion Matrix: {best_model_row["Model"]}\nThreshold={best_threshold:.3f}', 
          fontsize=14, fontweight='bold')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.savefig(classification_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ {classification_dir / 'confusion_matrix.png'}")
plt.close()

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
print("\n" + "=" * 90)
print(" ✅ PIPELINE COMPLETADO EXITOSAMENTE ".center(90, "="))
print("=" * 90)

print(f"\n📁 Archivos generados en: {classification_dir}")
print(f"   • model_comparison.csv")
print(f"   • test_predictions.csv")
print(f"   • final_metrics.csv")
print(f"   • confusion_matrix.csv")
print(f"   • confusion_matrix.png")
print(f"   • training_summary.txt")

print(f"\n🏆 Mejor modelo: {best_model_row['Model']}")
print(f"   F1-macro: {best_model_row['F1_macro_test']:.4f}")
print(f"   Accuracy: {best_model_row['Accuracy_test']:.4f}")
print(f"   Precision Low: {best_model_row['Precision_Low_test']:.4f}")
print(f"   Threshold: {best_threshold:.3f}")

print("\n" + "=" * 90)
