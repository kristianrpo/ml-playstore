import pandas as pd
import numpy as np
from pathlib import Path
from pipeline_data_preparation import GooglePlayDataPreparationPipeline
from pipeline_train_tunning import ModelTrainingPipeline


def load_data(path, file):
    """
    Carga datos desde una ruta relativa al repositorio o absoluta
    """
    # Resolve repository root (two levels above this script: notebooks/deliverable2 -> notebooks -> repo)
    repo_root = Path(__file__).resolve().parents[2]

    p = Path(path)
    # If the provided path is not absolute, interpret it relative to the repository root
    if not p.is_absolute():
        p = repo_root / path

    full_path = p / file

    # If the user accidentally passed the full file path in `path`, try that as a fallback
    alt_file = Path(path)
    if alt_file.is_file():
        return pd.read_csv(alt_file)

    if not full_path.exists():
        raise FileNotFoundError(
            f"File not found: {full_path}\n"
            f"Tried resolving relative to repo root: {repo_root}\n"
            "Provide an absolute path or a path relative to the repository root."
        )

    return pd.read_csv(full_path)


def prepare_data_for_training(train, val, test, target_col='Rating'):
    """
    Separa features y target de los datasets
    
    Args:
        train: DataFrame de entrenamiento
        val: DataFrame de validación
        test: DataFrame de test
        target_col: Nombre de la columna objetivo
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Work on copies to avoid mutating original dataframes
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Verificar que la columna target existe en todos los datasets
    for dataset_name, dataset in [('train', train), ('val', val), ('test', test)]:
        if target_col not in dataset.columns:
            raise ValueError(f'Target column "{target_col}" not found in {dataset_name} dataset')
    
    # Obtener todas las columnas numéricas de train
    numeric_columns = train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir la columna target y la columna 'App' si existe
    feature_columns = [col for col in numeric_columns 
                      if col != target_col and col != 'App']
    
    # Verificar que las columnas features existen en val y test
    common_features = [col for col in feature_columns 
                      if col in val.columns and col in test.columns]
    
    if len(common_features) == 0:
        raise ValueError('No common numeric features found in train/val/test. Revisa el preprocesamiento.')
    
    # Verificar que no hay columnas no numéricas en las features
    non_numeric_features = [col for col in common_features 
                           if not pd.api.types.is_numeric_dtype(train[col])]
    if non_numeric_features:
        raise ValueError(f'Non-numeric features found: {non_numeric_features}')
    
    # Crear X e y para cada dataset
    X_train = train[common_features]
    y_train = train[target_col]

    X_val = val[common_features]
    y_val = val[target_col]

    X_test = test[common_features]
    y_test = test[target_col]
    
    # Verificaciones finales
    print(f"Features seleccionadas ({len(common_features)}): {common_features}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Target column: {target_col}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    """
    Pipeline completo: Preparación de datos + Entrenamiento de modelos
    """
    
    print("=" * 80)
    print(" PIPELINE COMPLETO: PREPARACIÓN + ENTRENAMIENTO ".center(80))
    print("=" * 80)
    
    # ========================================================================
    # PASO 1: CARGAR DATOS
    # ========================================================================
    print("\n[PASO 1/3] Cargando datos originales...")
    print("-" * 80)
    
    df = load_data("./data/original/google-play-store", "googleplaystore.csv")
    print(f"✓ Datos cargados: {df.shape}")
    print(f"  Columnas: {list(df.columns)}")
    
    # ========================================================================
    # PASO 2: PREPARACIÓN DE DATOS
    # ========================================================================
    print("\n[PASO 2/3] Ejecutando pipeline de preparación de datos...")
    print("-" * 80)
    
    data_pipeline = GooglePlayDataPreparationPipeline(
        test_size=0.30,
        val_size=0.50,
        category_threshold=70,
        mi_threshold=0.01,
        corr_threshold=0.8,
        vars_to_remove=['Installs_log'],
        reference_date='2025-10-02',
        random_state=42,
        verbose=True,
        plot=False  # Cambiar a True para ver visualizaciones
    )
    
    # Ejecutar pipeline de preparación
    train, val, test = data_pipeline.fit_transform(df)
    
    
    # Guardar datasets procesados
    print("\nGuardando datasets procesados...")
    # Save inside the deliverable2 outputs folder
    outputs_dir = Path(__file__).resolve().parent / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_fp = outputs_dir / 'train_processed.csv'
    val_fp = outputs_dir / 'val_processed.csv'
    test_fp = outputs_dir / 'test_processed.csv'

    train.to_csv(train_fp, index=False)
    val.to_csv(val_fp, index=False)
    test.to_csv(test_fp, index=False)
    print(f"   ✓ {train_fp}")
    print(f"   ✓ {val_fp}")
    print(f"   ✓ {test_fp}")
    
    # ========================================================================
    # PASO 3: ENTRENAMIENTO DE MODELOS
    # ========================================================================
    print("\n[PASO 3/3] Entrenando modelos con orquestador...")
    print("-" * 80)
    
    # Separar features y target
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(
        train, val, test, target_col='Rating'
    )
    
    print(f"\nDimensiones de entrenamiento:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   Features: {list(X_train.columns)}")
    
    # Inicializar orquestador de entrenamiento
    # Opción 1: Entrenar todas las familias (por defecto)
    # training_pipeline = ModelTrainingPipeline(
    #     cv_folds=3,
    #     random_state=42,
    #     n_jobs=-1,
    #     verbose=True
    # )
    
    # Opción 2: Entrenar todas las familias (recomendado para mejor rendimiento)
    training_pipeline = ModelTrainingPipeline(
        cv_folds=5,  # Aumentar CV folds para mejor validación
        random_state=42,
        n_jobs=-1,
        verbose=True,
        families=['trees']
    )
    
    # Otras opciones:
    # families=['svm'] - Solo SVM
    # families=['trees'] - Solo árboles
    # families=['boosting'] - Solo boosting
    # families=['svm', 'trees'] - SVM y árboles
    # families=['svm', 'boosting'] - SVM y boosting
    
    # Entrenar todos los modelos (SVM, Árboles, Boosting)
    training_pipeline.fit(X_train, y_train, X_val, y_val)
    
    # ========================================================================
    # PASO 4: EVALUACIÓN EN TEST
    # ========================================================================
    print("\n" + "=" * 80)
    print(" EVALUACIÓN EN CONJUNTO DE TEST ".center(80))
    print("=" * 80)
    
    # Obtener mejor modelo
    best_model, best_info = training_pipeline.get_best_model()
    
    print(f"\nMejor modelo: {best_info['Modelo']} ({best_info['Familia']})")
    print(f"   MAE (val):  {best_info['MAE_val']:.4f}")
    print(f"   RMSE (val): {best_info['RMSE_val']:.4f}")
    print(f"   R² (val):   {best_info['R2_val']:.4f}")
    
    # Predicciones en test
    y_pred_test = training_pipeline.predict(X_test, family='best')
    
    # Calcular métricas en test
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nMétricas en TEST:")
    print(f"   MAE:  {mae_test:.4f}")
    print(f"   RMSE: {rmse_test:.4f}")
    print(f"   R²:   {r2_test:.4f}")
    
    # Comparar val vs test
    print(f"\nComparación Val vs Test:")
    print(f"   MAE:  {abs(best_info['MAE_val'] - mae_test):.4f}")
    print(f"   RMSE: {abs(best_info['RMSE_val'] - rmse_test):.4f}")
    print(f"   R²:   {abs(best_info['R2_val'] - r2_test):.4f}")
    
    # ========================================================================
    # PASO 5: GUARDAR RESULTADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print(" GUARDANDO RESULTADOS ".center(80))
    print("=" * 80)
    
    # Guardar comparación y resumen en la carpeta de outputs del deliverable
    training_pipeline.save_results(output_dir=outputs_dir)
    
    # Guardar predicciones de test
    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred_test,
        'error': y_test - y_pred_test,
        'abs_error': np.abs(y_test - y_pred_test)
    })
    test_predictions_fp = outputs_dir / 'test_predictions.csv'
    test_results.to_csv(test_predictions_fp, index=False)
    print(f"   ✓ {test_predictions_fp}")
    
    # Guardar métricas finales
    final_metrics = pd.DataFrame({
        'Dataset': ['Train', 'Val', 'Test'],
        'MAE': [best_info['MAE_train'], best_info['MAE_val'], mae_test],
        'RMSE': [None, best_info['RMSE_val'], rmse_test],
        'R2': [None, best_info['R2_val'], r2_test]
    })
    final_metrics_fp = outputs_dir / 'final_metrics.csv'
    final_metrics.to_csv(final_metrics_fp, index=False)
    print(f"   ✓ {final_metrics_fp}")
    
    # Feature importance (si está disponible)
    feature_importance = training_pipeline.get_feature_importance(top_n=20)
    if feature_importance is not None:
        feature_importance_fp = outputs_dir / 'feature_importance.csv'
        feature_importance.to_csv(feature_importance_fp, index=False)
        print(f"   ✓ {feature_importance_fp}")
        
        print(f"\nTop 10 Features más importantes:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETADO EXITOSAMENTE ".center(80, "="))
    print("=" * 80)
    
    print(f"\nArchivos generados:")
    print("   Datos procesados:")
    print(f"      - {train_fp}")
    print(f"      - {val_fp}")
    print(f"      - {test_fp}")
    print("   Resultados de modelos:")
    print(f"      - {outputs_dir / 'model_comparison.csv'}")
    print(f"      - {outputs_dir / 'training_summary.txt'}")
    print(f"      - {test_predictions_fp}")
    print(f"      - {final_metrics_fp}")
    if feature_importance is not None:
        print(f"      - {feature_importance_fp}")
    
    print(f"\nMejor modelo: {best_info['Modelo']}")
    print(f"   MAE en test: {mae_test:.4f}")
    print(f"   R² en test:  {r2_test:.4f}")
    
    print("\n" + "=" * 80)