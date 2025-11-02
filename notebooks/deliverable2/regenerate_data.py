"""
Script para regenerar datos procesados con el pipeline actualizado.

CAMBIOS IMPLEMENTADOS:
1. ✅ Genres procesado (primer género extraído) - NO eliminado
2. ✅ category_threshold reducido de 70 a 30 (más categorías)
3. ✅ mi_threshold más permisivo: 0.001 (antes 0.005)
4. ✅ MutualInfoSelector DESACTIVADO (mantiene todas las features)

Esto debería aumentar el número de features de ~62 a ~80-100
y mejorar la accuracy de 41% a 55-65%.
"""

import pandas as pd
from pipeline_data_preparation import GooglePlayDataPreparationPipeline

# Configuración
DATA_PATH = '../../data/original/google-play-store/googleplaystore.csv'
OUTPUT_DIR = './outputs'

print("=" * 80)
print("REGENERANDO DATOS CON PIPELINE ACTUALIZADO".center(80))
print("=" * 80)

# Cargar datos crudos
print("\n[1/3] Cargando datos crudos...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Datos cargados: {df.shape}")

# Ejecutar pipeline con nuevos parámetros
print("\n[2/3] Ejecutando pipeline mejorado...")
print("\nCAMBIOS:")
print("  • Genres: INCLUIDO (extraído primer género)")
print("  • category_threshold: 30 (antes 70) → MÁS categorías")
print("  • mi_threshold: 0.001 (antes 0.005) → MÁS permisivo")
print("  • MutualInfoSelector: DESACTIVADO → SIN eliminación de features")
print("=" * 80)

pipeline = GooglePlayDataPreparationPipeline(
    test_size=0.30,
    val_size=0.50,
    category_threshold=30,      # REDUCIDO
    mi_threshold=0.001,          # MÁS PERMISIVO (pero no se usa)
    corr_threshold=0.8,
    vars_to_remove=['Installs_log'],
    reference_date='2025-10-02',
    random_state=42,
    verbose=True,
    plot=False
)

train, val, test = pipeline.fit_transform(df)

# Guardar datasets procesados
print("\n[3/3] Guardando datasets procesados...")
train.to_csv(f'{OUTPUT_DIR}/train_processed.csv', index=False)
val.to_csv(f'{OUTPUT_DIR}/val_processed.csv', index=False)
test.to_csv(f'{OUTPUT_DIR}/test_processed.csv', index=False)

print("\n✓ Datasets guardados:")
print(f"  {OUTPUT_DIR}/train_processed.csv: {train.shape}")
print(f"  {OUTPUT_DIR}/val_processed.csv: {val.shape}")
print(f"  {OUTPUT_DIR}/test_processed.csv: {test.shape}")

# Resumen de cambios
print("\n" + "=" * 80)
print("RESUMEN DE FEATURES".center(80))
print("=" * 80)

# Comparar con versión anterior (62 features)
print(f"\n✓ Features ANTES: ~62")
print(f"✓ Features AHORA: {train.shape[1]}")
print(f"✓ Incremento: +{train.shape[1] - 62} features")

# Mostrar features categóricas
cat_features = [col for col in train.columns if '_' in col and col not in ['size_missing', 'Size_log']]
print(f"\n✓ Features categóricas (dummies): {len(cat_features)}")

# Verificar si Genres está presente
genres_features = [col for col in train.columns if col.startswith('Genres_')]
if genres_features:
    print(f"\n✅ Genres INCLUIDO: {len(genres_features)} categorías")
    print(f"   Ejemplos: {genres_features[:5]}")
else:
    print("\n⚠️  WARNING: Genres NO encontrado en las features")

# Verificar categorías
category_features = [col for col in train.columns if col.startswith('Category_')]
print(f"\n✓ Category: {len(category_features)} categorías (antes ~20)")

print("\n" + "=" * 80)
print("✅ REGENERACIÓN COMPLETADA".center(80))
print("=" * 80)
print("\n📊 Próximo paso: Ejecutar main_classification.py")
print("   Accuracy esperada: 55-65% (antes 41%)")
