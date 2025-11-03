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
        n_bins=2,  # 2 categorías: Low (≤4.3) y High (>4.3)
        binning_strategy='custom',  # Usar umbral personalizado en 4.0
        cv_folds=5,
        random_state=42,
        n_jobs=-1,
        verbose=True,
        families=['boosting', 'trees', 'knn'],  # Solo árboles y boosting (más rápido)
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

    # Calibración de umbral (solo binaria): optimiza F1_macro en validación
    calibrated_threshold = None
    calibrated_metrics = None
    y_pred_test_calibrated = None
    if len(classification_pipeline.target_transformer.labels) == 2:
        # Probabilidades en validación para la clase 1 (High)
        y_proba_val = classification_pipeline.predict_proba(X_val, family='best')
        proba_high_val = y_proba_val[:, 1]

        def sweep_best_threshold(proba, y_true, metric='f1_macro'):
            thresholds = np.linspace(0.3, 0.7, 21)
            best_t, best_score = 0.5, -1.0
            for t in thresholds:
                y_pred_t = (proba >= t).astype(int)
                score = f1_score(y_true, y_pred_t, average='macro', zero_division=0)
                if score > best_score:
                    best_score, best_t = score, t
            return best_t, best_score

        calibrated_threshold, best_val_macro = sweep_best_threshold(proba_high_val, y_val_cat)

        # Aplicar umbral calibrado a test
        proba_high_test = y_proba_test[:, 1]
        y_pred_test_calibrated = (proba_high_test >= calibrated_threshold).astype(int)

        # Métricas calibradas en val/test
        accuracy_val_cal = accuracy_score(y_val_cat, (proba_high_val >= calibrated_threshold).astype(int))
        precision_val_cal = precision_score(y_val_cat, (proba_high_val >= calibrated_threshold).astype(int), average='weighted', zero_division=0)
        recall_val_cal = recall_score(y_val_cat, (proba_high_val >= calibrated_threshold).astype(int), average='weighted', zero_division=0)
        f1_val_cal = f1_score(y_val_cat, (proba_high_val >= calibrated_threshold).astype(int), average='weighted', zero_division=0)
        f1_macro_val_cal = f1_score(y_val_cat, (proba_high_val >= calibrated_threshold).astype(int), average='macro', zero_division=0)

        accuracy_test_cal = accuracy_score(y_test_cat, y_pred_test_calibrated)
        precision_test_cal = precision_score(y_test_cat, y_pred_test_calibrated, average='weighted', zero_division=0)
        recall_test_cal = recall_score(y_test_cat, y_pred_test_calibrated, average='weighted', zero_division=0)
        f1_test_cal = f1_score(y_test_cat, y_pred_test_calibrated, average='weighted', zero_division=0)
        f1_macro_test_cal = f1_score(y_test_cat, y_pred_test_calibrated, average='macro', zero_division=0)

        calibrated_metrics = {
            'threshold': calibrated_threshold,
            'val': {
                'Accuracy': accuracy_val_cal,
                'Precision': precision_val_cal,
                'Recall': recall_val_cal,
                'F1_weighted': f1_val_cal,
                'F1_macro': f1_macro_val_cal
            },
            'test': {
                'Accuracy': accuracy_test_cal,
                'Precision': precision_test_cal,
                'Recall': recall_test_cal,
                'F1_weighted': f1_test_cal,
                'F1_macro': f1_macro_test_cal
            }
        }
    
    # Calcular métricas en test
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

    if calibrated_metrics is not None:
        print("\n🎯 Umbral calibrado (validación, F1_macro):")
        print(f"   Threshold óptimo: {calibrated_metrics['threshold']:.3f}")
        print(f"   Val F1_macro (calibrado): {calibrated_metrics['val']['F1_macro']:.4f}")
        print(f"   Test F1_macro (calibrado): {calibrated_metrics['test']['F1_macro']:.4f}")
    
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
    
    # Matriz de confusión
    print("\nGenerando matriz de confusión...")
    classification_pipeline.plot_confusion_matrix(X_test, y_test_cat, family='best', normalize=True)
    
    # Classification report
    print("\nReporte de clasificación completo:")
    print(classification_pipeline.get_classification_report(X_test, y_test_cat, family='best'))
    
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

    # Guardar predicciones calibradas (si aplica)
    if y_pred_test_calibrated is not None:
        test_results_cal = pd.DataFrame({
            'y_true': y_test_cat,
            'y_pred': y_pred_test_calibrated,
            'correct': (y_test_cat == y_pred_test_calibrated).astype(int)
        })
        for i, label in enumerate(classification_pipeline.target_transformer.labels):
            test_results_cal[f'proba_{label}'] = y_proba_test[:, i]
        test_predictions_cal_fp = outputs_dir / 'test_predictions_calibrated.csv'
        test_results_cal.to_csv(test_predictions_cal_fp, index=False)
        print(f"   ✓ {test_predictions_cal_fp}")
    
    # Guardar métricas finales
    rows = [
        {'Dataset': 'Val', 'Accuracy': best_info['Accuracy_val'], 'Precision': best_info['Precision_val'], 'Recall': best_info['Recall_val'], 'F1_weighted': best_info['F1_weighted_val'], 'F1_macro': best_info['F1_macro_val']},
        {'Dataset': 'Test', 'Accuracy': accuracy_test, 'Precision': precision_test, 'Recall': recall_test, 'F1_weighted': f1_test, 'F1_macro': f1_macro_test}
    ]
    if calibrated_metrics is not None:
        rows.append({'Dataset': 'Val_calibrated', **calibrated_metrics['val']})
        rows.append({'Dataset': 'Test_calibrated', **calibrated_metrics['test']})
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

    # Guardar umbral calibrado
    if calibrated_metrics is not None:
        with open(outputs_dir / 'threshold_calibration.txt', 'w', encoding='utf-8') as f:
            f.write(f"Threshold óptimo (Val, F1_macro): {calibrated_metrics['threshold']:.4f}\n")
            f.write("Métricas calibradas (Val):\n")
            for k, v in calibrated_metrics['val'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("Métricas calibradas (Test):\n")
            for k, v in calibrated_metrics['test'].items():
                f.write(f"  {k}: {v:.4f}\n")
    
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