"""
Main script para ejecutar el pipeline de clasificación de ratings de Google Play Store.

Este script:
1. Carga los datos procesados (train/val/test)
2. Ejecuta el pipeline de clasificación con todas las familias de modelos
3. Evalúa el mejor modelo en el conjunto de test
4. Genera visualizaciones y reportes
5. Guarda todos los resultados

Uso:
    python main_classification.py

Autor: Sistema de ML
Fecha: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Importar pipeline de clasificación
from pipeline_classification import ClassificationTrainingPipeline

# Importar métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(data_dir: str = '../../data/original/google-play-store'):
    """
    Carga los datos procesados desde CSVs.
    
    Args:
        data_dir: Directorio con los datos procesados
        
    Returns:
        tuple: (train, val, test) DataFrames
    """
    data_dir = Path(data_dir)
    
    # Intentar cargar desde outputs primero
    outputs_dir = Path(__file__).resolve().parent / 'outputs'
    
    if (outputs_dir / 'train_processed.csv').exists():
        print(f"Cargando datos desde: {outputs_dir}")
        train = pd.read_csv(outputs_dir / 'train_processed.csv')
        val = pd.read_csv(outputs_dir / 'val_processed.csv')
        test = pd.read_csv(outputs_dir / 'test_processed.csv')
    else:
        # Fallback: cargar desde data_dir
        print(f"Cargando datos desde: {data_dir}")
        train = pd.read_csv(data_dir / 'train_processed.csv')
        val = pd.read_csv(data_dir / 'val_processed.csv')
        test = pd.read_csv(data_dir / 'test_processed.csv')
    
    print(f"✓ Datos cargados:")
    print(f"  Train: {train.shape}")
    print(f"  Val:   {val.shape}")
    print(f"  Test:  {test.shape}")
    
    return train, val, test


def prepare_data_for_classification(train, val, test, target_col='Rating'):
    """
    Separa features y target de los datasets.
    
    Args:
        train: DataFrame de entrenamiento
        val: DataFrame de validación
        test: DataFrame de test
        target_col: Nombre de la columna objetivo
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Verificar que la columna target existe
    for dataset_name, dataset in [('train', train), ('val', val), ('test', test)]:
        if target_col not in dataset.columns:
            raise ValueError(f'Target column "{target_col}" not found in {dataset_name} dataset')
    
    # Obtener todas las columnas numéricas
    numeric_columns = train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir la columna target y 'App' si existe
    feature_columns = [col for col in numeric_columns 
                      if col != target_col and col != 'App']
    
    # Verificar que las columnas features existen en val y test
    common_features = [col for col in feature_columns 
                      if col in val.columns and col in test.columns]
    
    if len(common_features) == 0:
        raise ValueError('No common numeric features found in train/val/test')
    
    # Crear X e y para cada dataset
    X_train = train[common_features]
    y_train = train[target_col]

    X_val = val[common_features]
    y_val = val[target_col]

    X_test = test[common_features]
    y_test = test[target_col]
    
    print(f"\n✓ Features seleccionadas ({len(common_features)}):")
    print(f"  Primeras 10: {common_features[:10]}")
    print(f"\n✓ Dimensiones:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    """
    Pipeline completo: Clasificación de ratings
    """
    
    print("=" * 80)
    print(" PIPELINE DE CLASIFICACIÓN: GOOGLE PLAY STORE ".center(80))
    print("=" * 80)
    
    # ========================================================================
    # PASO 1: CARGAR DATOS PROCESADOS
    # ========================================================================
    print("\n[PASO 1/4] Cargando datos procesados...")
    print("-" * 80)
    
    try:
        train, val, test = load_processed_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: No se encontraron los datos procesados")
        print(f"   Ejecutar primero: python main.py")
        print(f"   O verificar la ruta de los archivos CSV")
        sys.exit(1)
    
    # ========================================================================
    # PASO 2: PREPARAR DATOS PARA CLASIFICACIÓN
    # ========================================================================
    print("\n[PASO 2/4] Preparando datos para clasificación...")
    print("-" * 80)
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_classification(
        train, val, test, target_col='Rating'
    )
    
    # ========================================================================
    # PASO 3: ENTRENAR MODELOS DE CLASIFICACIÓN
    # ========================================================================
    print("\n[PASO 3/4] Entrenando modelos de clasificación...")
    print("-" * 80)
    
    # Inicializar pipeline
    # Clasificación binaria: Rating > 4.0 (High) vs Rating ≤ 4.0 (Low)
    classification_pipeline = ClassificationTrainingPipeline(
        n_bins=2,  # 2 categorías: Low (≤4.0) y High (>4.0)
        binning_strategy='custom',  # Usar umbral personalizado en 4.0
        cv_folds=5,
        random_state=42,
        n_jobs=-1,
        verbose=True,
        families=['trees','boosting'],  # Boosting y Trees (sin KNN por overfitting)
        custom_bins=[0, 4, 5.1],  # Umbral en 4.0
        custom_labels=['Low', 'High']  # Etiquetas personalizadas
    )
    
    # Entrenar todos los modelos
    classification_pipeline.fit(X_train, y_train, X_val, y_val)
    
    # ========================================================================
    # PASO 4: EVALUACIÓN EN TEST
    # ========================================================================
    print("\n" + "=" * 80)
    print(" EVALUACIÓN EN CONJUNTO DE TEST ".center(80))
    print("=" * 80)
    
    # Obtener mejor modelo
    best_model, best_info = classification_pipeline.get_best_model()
    
    print(f"\n🏆 Mejor modelo: {best_info['Modelo']} ({best_info['Familia']})")
    print(f"   Accuracy (val):  {best_info['Accuracy_val']:.4f}")
    print(f"   F1-score (val):  {best_info['F1_weighted_val']:.4f}")
    print(f"   Precision (val): {best_info['Precision_val']:.4f}")
    print(f"   Recall (val):    {best_info['Recall_val']:.4f}")
    
    # Transformar target de test a categorías
    y_test_cat = classification_pipeline.target_transformer.transform(y_test)
    # Transformar target de validación a categorías (para calibración de umbral)
    y_val_cat = classification_pipeline.target_transformer.transform(y_val)
    
    # Predicciones en test
    y_pred_test = classification_pipeline.predict(X_test, family='best')
    y_proba_test = classification_pipeline.predict_proba(X_test, family='best')
    
    # ========================================================================
    # CALIBRACIÓN DE THRESHOLD (para mejorar balance Low/High)
    # ========================================================================
    print("\n" + "=" * 80)
    print(" CALIBRACIÓN DE THRESHOLD (Class Weights + Threshold) ".center(80))
    print("=" * 80)
    
    # Obtener probabilidades en validación
    y_proba_val = classification_pipeline.predict_proba(X_val, family='best')
    
    # Buscar el mejor threshold optimizando F1-macro (promedio de ambas clases)
    print("\n🎯 Buscando threshold óptimo para maximizar F1-macro (promedio de ambas clases)...")
    
    from sklearn.metrics import precision_recall_fscore_support
    
    thresholds = np.arange(0.3, 0.8, 0.01)  # Rango de thresholds a probar
    best_threshold = 0.5
    best_f1_macro = 0.0
    
    results = []
    for thresh in thresholds:
        # Predicciones con este threshold (clase 1 = High si proba >= thresh)
        y_pred_val_thresh = (y_proba_val[:, 1] >= thresh).astype(int)
        
        # Calcular F1 por clase
        p, r, f, _ = precision_recall_fscore_support(y_val_cat, y_pred_val_thresh, labels=[0, 1], zero_division=0)
        
        # F1-macro es el promedio de F1 de ambas clases
        f1_macro_current = (f[0] + f[1]) / 2
        
        # Guardamos resultados
        results.append({
            'threshold': thresh,
            'f1_low': f[0],
            'precision_low': p[0],
            'recall_low': r[0],
            'f1_high': f[1],
            'f1_macro': f1_macro_current
        })
        
        # Actualizar mejor threshold si mejora F1-macro
        if f1_macro_current > best_f1_macro:
            best_f1_macro = f1_macro_current
            best_threshold = thresh
    
    print(f"   Threshold por defecto: 0.50")
    print(f"   Threshold óptimo: {best_threshold:.3f}")
    print(f"   F1-macro (val): {best_f1_macro:.4f}")
    
    # Aplicar el threshold óptimo a validación y test
    y_pred_val_calibrated = (y_proba_val[:, 1] >= best_threshold).astype(int)
    y_pred_test_calibrated = (y_proba_test[:, 1] >= best_threshold).astype(int)
    
    # Métricas con threshold calibrado
    print("\n📊 Comparación Threshold 0.5 vs Calibrado:")
    print("\nValidación:")
    
    # Métricas con threshold 0.5
    y_pred_val_default = classification_pipeline.predict(X_val, family='best')
    p_val_def, r_val_def, f_val_def, _ = precision_recall_fscore_support(y_val_cat, y_pred_val_default, labels=[0, 1], zero_division=0)
    
    # Métricas con threshold calibrado
    p_val_cal, r_val_cal, f_val_cal, _ = precision_recall_fscore_support(y_val_cat, y_pred_val_calibrated, labels=[0, 1], zero_division=0)
    
    print(f"   Low  - F1: {f_val_def[0]:.4f} → {f_val_cal[0]:.4f} (Δ={f_val_cal[0]-f_val_def[0]:+.4f})")
    print(f"   Low  - Precision: {p_val_def[0]:.4f} → {p_val_cal[0]:.4f} (Δ={p_val_cal[0]-p_val_def[0]:+.4f})")
    print(f"   Low  - Recall: {r_val_def[0]:.4f} → {r_val_cal[0]:.4f} (Δ={r_val_cal[0]-r_val_def[0]:+.4f})")
    print(f"   High - F1: {f_val_def[1]:.4f} → {f_val_cal[1]:.4f} (Δ={f_val_cal[1]-f_val_def[1]:+.4f})")
    
    print("\nTest:")
    # Métricas en test con threshold calibrado
    p_test_cal, r_test_cal, f_test_cal, _ = precision_recall_fscore_support(y_test_cat, y_pred_test_calibrated, labels=[0, 1], zero_division=0)
    p_test_def, r_test_def, f_test_def, _ = precision_recall_fscore_support(y_test_cat, y_pred_test, labels=[0, 1], zero_division=0)
    
    print(f"   Low  - F1: {f_test_def[0]:.4f} → {f_test_cal[0]:.4f} (Δ={f_test_cal[0]-f_test_def[0]:+.4f})")
    print(f"   Low  - Precision: {p_test_def[0]:.4f} → {p_test_cal[0]:.4f} (Δ={p_test_cal[0]-p_test_def[0]:+.4f})")
    print(f"   Low  - Recall: {r_test_def[0]:.4f} → {r_test_cal[0]:.4f} (Δ={r_test_cal[0]-r_test_def[0]:+.4f})")
    print(f"   High - F1: {f_test_def[1]:.4f} → {f_test_cal[1]:.4f} (Δ={f_test_cal[1]-f_test_def[1]:+.4f})")
    
    # Usar predicciones calibradas para el resto del análisis
    y_pred_test = y_pred_test_calibrated
    
    # ========================================================================
    
    # Calcular métricas en test (con threshold calibrado)
    accuracy_test = accuracy_score(y_test_cat, y_pred_test)
    precision_test = precision_score(y_test_cat, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test_cat, y_pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(y_test_cat, y_pred_test, average='weighted', zero_division=0)
    f1_macro_test = f1_score(y_test_cat, y_pred_test, average='macro', zero_division=0)
    
    print(f"\n📊 Métricas en TEST:")
    print(f"   Accuracy:        {accuracy_test:.4f}")
    print(f"   Precision:       {precision_test:.4f}")
    print(f"   Recall:          {recall_test:.4f}")
    print(f"   F1-score (weighted): {f1_test:.4f}")
    print(f"   F1-score (macro):    {f1_macro_test:.4f}")
    
    # Comparar val vs test
    print(f"\n📈 Comparación Val vs Test:")
    print(f"   Accuracy:  Δ = {abs(best_info['Accuracy_val'] - accuracy_test):.4f}")
    print(f"   F1-score:  Δ = {abs(best_info['F1_weighted_val'] - f1_test):.4f}")
    
    # ========================================================================
    # PASO 5: VISUALIZACIONES
    # ========================================================================
    print("\n" + "=" * 80)
    print(" GENERANDO VISUALIZACIONES ".center(80))
    print("=" * 80)
    
    # Matriz de confusión (usando predicciones calibradas)
    print("\nGenerando matriz de confusión...")
    cm = confusion_matrix(y_test_cat, y_pred_test)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualizar
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f',
               cmap='Blues', 
               xticklabels=classification_pipeline.target_transformer.labels,
               yticklabels=classification_pipeline.target_transformer.labels)
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.title(f'Matriz de Confusión (Test) - {best_info["Modelo"]} (Threshold={best_threshold:.3f})')
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("\nReporte de clasificación completo:")
    print(classification_report(y_test_cat, y_pred_test,
                              target_names=classification_pipeline.target_transformer.labels,
                              zero_division=0))
    
    # ========================================================================
    # PASO 6: GUARDAR RESULTADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print(" GUARDANDO RESULTADOS ".center(80))
    print("=" * 80)
    
    # Crear directorio de salida
    outputs_dir = Path(__file__).resolve().parent / 'outputs' / 'classification'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar comparación y resumen
    classification_pipeline.save_results(output_dir=outputs_dir)
    
    # Guardar predicciones de test
    test_results = pd.DataFrame({
        'y_true': y_test_cat,
        'y_pred': y_pred_test,
        'correct': (y_test_cat == y_pred_test).astype(int)
    })
    
    # Añadir probabilidades por clase
    for i, label in enumerate(classification_pipeline.target_transformer.labels):
        test_results[f'proba_{label}'] = y_proba_test[:, i]
    
    test_predictions_fp = outputs_dir / 'test_predictions.csv'
    test_results.to_csv(test_predictions_fp, index=False)
    print(f"   ✓ {test_predictions_fp}")
    
    # Guardar métricas finales
    rows = [
        {'Dataset': 'Val', 'Accuracy': best_info['Accuracy_val'], 'Precision': best_info['Precision_val'], 'Recall': best_info['Recall_val'], 'F1_weighted': best_info['F1_weighted_val'], 'F1_macro': best_info['F1_macro_val']},
        {'Dataset': 'Test', 'Accuracy': accuracy_test, 'Precision': precision_test, 'Recall': recall_test, 'F1_weighted': f1_test, 'F1_macro': f1_macro_test}
    ]
    final_metrics = pd.DataFrame(rows)
    final_metrics_fp = outputs_dir / 'final_metrics.csv'
    final_metrics.to_csv(final_metrics_fp, index=False)
    print(f"   ✓ {final_metrics_fp}")
    
    # Guardar matriz de confusión
    cm = confusion_matrix(y_test_cat, y_pred_test)
    cm_df = pd.DataFrame(
        cm,
        index=classification_pipeline.target_transformer.labels,
        columns=classification_pipeline.target_transformer.labels
    )
    cm_fp = outputs_dir / 'confusion_matrix.csv'
    cm_df.to_csv(cm_fp)
    print(f"   ✓ {cm_fp}")
    
    # Feature importance (si está disponible)
    if best_info['Familia'] == 'Árboles':
        feature_importance = classification_pipeline.tree_family.get_feature_importance(
            model_name=best_info['Modelo'],
            top_n=20
        )
        if feature_importance is not None:
            feature_importance_fp = outputs_dir / 'feature_importance.csv'
            feature_importance.to_csv(feature_importance_fp, index=False)
            print(f"   ✓ {feature_importance_fp}")
            
            print(f"\n📊 Top 10 Features más importantes:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETADO EXITOSAMENTE ".center(80, "="))
    print("=" * 80)
    
    print(f"\n📁 Archivos generados:")
    print(f"   Resultados de modelos:")
    print(f"      - {outputs_dir / 'classification_model_comparison.csv'}")
    print(f"      - {outputs_dir / 'classification_training_summary.txt'}")
    print(f"      - {test_predictions_fp}")
    print(f"      - {final_metrics_fp}")
    print(f"      - {cm_fp}")
    if best_info['Familia'] == 'Árboles':
        print(f"      - {outputs_dir / 'feature_importance.csv'}")
    
    print(f"\n🏆 Mejor modelo: {best_info['Modelo']}")
    print(f"   Accuracy en test:  {accuracy_test:.4f}")
    print(f"   F1-score en test:  {f1_test:.4f}")
    
    print(f"\n📊 Bins de Rating utilizados:")
    for i, edge in enumerate(classification_pipeline.target_transformer.get_bin_edges()[:-1]):
        next_edge = classification_pipeline.target_transformer.get_bin_edges()[i+1]
        label = classification_pipeline.target_transformer.labels[i]
        print(f"   {label}: [{edge:.3f}, {next_edge:.3f})")
    
    print("\n" + "=" * 80)