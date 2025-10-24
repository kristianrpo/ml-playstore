
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TRANSFORMADOR 1: CONVERSIÓN DE COLUMNAS NUMÉRICAS
# ============================================================================

class NumericConverter(BaseEstimator, TransformerMixin):
    """
    Convierte columnas a numéricas siguiendo la lógica del proyecto:
    - Rating: directamente a numérico
    - Reviews: elimina comas y caracteres no numéricos
    - Installs: elimina +, comas y crea columna 'Installs Numeric'
    - Price: elimina $ y caracteres no numéricos
    - Size: convierte M y k a MB numérico
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("=" * 80)
            print("PASO 0: CONVERSIÓN DE COLUMNAS NUMÉRICAS".center(80))
            print("=" * 80)
        
        # Rating
        if "Rating" in df.columns:
            df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
            if self.verbose:
                print(f"✓ Rating convertido a numérico")
        
        # Reviews: quitar comas y cualquier carácter no numérico/punto
        if "Reviews" in df.columns:
            df["Reviews"] = (
                df["Reviews"].astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if self.verbose:
                print(f"✓ Reviews convertido a numérico")
        
        # Installs: quitar +, comas y cualquier carácter no numérico/punto
        if "Installs" in df.columns:
            df["Installs Numeric"] = (
                df["Installs"].astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if self.verbose:
                print(f"✓ Installs convertido a numérico (columna 'Installs Numeric')")
        
        # Price: quitar $ y cualquier carácter no numérico/punto
        if "Price" in df.columns:
            df["Price"] = (
                df["Price"].astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
            if self.verbose:
                print(f"✓ Price convertido a numérico")
        
        # Size: convertir M y k/K a MB
        if "Size" in df.columns:
            def parse_size(x):
                if isinstance(x, str):
                    x = x.strip()
                    if x.endswith("M"):
                        return float(x[:-1])
                    elif x.endswith("k") or x.endswith("K"):
                        return float(x[:-1]) / 1024  # KB -> MB
                    else:
                        return np.nan
                return np.nan
            
            df["Size"] = df["Size"].apply(parse_size)
            if self.verbose:
                print(f"✓ Size convertido a MB numérico")
        
        if self.verbose:
            print(f"\nTotal de registros: {len(df):,}")
        
        return df


# ============================================================================
# TRANSFORMADOR 2: ELIMINACIÓN DE DUPLICADOS
# ============================================================================

class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Elimina filas duplicadas del DataFrame, manteniendo la primera ocurrencia.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.initial_size = None
        self.duplicates_found = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 1: ELIMINACIÓN DE DUPLICADOS".center(80))
            print("=" * 80)
        
        # Estado inicial
        self.initial_size = len(df)
        self.duplicates_found = df.duplicated().sum()
        
        if self.verbose:
            print(f"\nRegistros antes de eliminar duplicados: {self.initial_size:,}")
            print(f"Duplicados encontrados: {self.duplicates_found:,} ({self.duplicates_found/self.initial_size*100:.2f}%)")
        
        # Eliminar duplicados (manteniendo la primera ocurrencia)
        df = df.drop_duplicates(keep='first')
        
        # Estado final
        if self.verbose:
            print(f"Registros después de eliminar duplicados: {len(df):,}")
            print(f"Filas eliminadas: {self.initial_size - len(df):,}")
            print(f"Reducción: {((self.initial_size - len(df))/self.initial_size*100):.2f}%")
        
        return df


# ============================================================================
# TRANSFORMADOR 3: CORRECCIÓN DE VALORES IMPOSIBLES
# ============================================================================

class ImpossibleValuesRemover(BaseEstimator, TransformerMixin):
    """
    Elimina registros con valores imposibles:
    - Rating fuera del rango [1, 5]
    - Valores negativos en Reviews, Size, Price, Installs Numeric
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.removed_counts = {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 2: CORRECCIÓN DE VALORES IMPOSIBLES".center(80))
            print("=" * 80)
        
        initial_len = len(df)
        
        # Verificar valores de Rating fuera del rango [1, 5]
        if 'Rating' in df.columns:
            invalid_ratings = ((df['Rating'] < 1) | (df['Rating'] > 5))
            invalid_count = invalid_ratings.sum()
            
            if self.verbose:
                print(f"\nRatings inválidos encontrados: {invalid_count}")
            
            if invalid_count > 0:
                if self.verbose:
                    invalid_values = sorted(df.loc[invalid_ratings, 'Rating'].dropna().unique())
                    print(f"Valores inválidos únicos: {invalid_values}")
                
                # Eliminar registros con valores inválidos
                df = df[~invalid_ratings].copy()
                self.removed_counts['rating'] = invalid_count
                
                if self.verbose:
                    print(f"✓ {invalid_count} registros con Rating inválido eliminados")
            else:
                if self.verbose:
                    print("✓ No se encontraron ratings inválidos")
        
        # Verificar valores negativos en columnas numéricas
        if self.verbose:
            print("\n" + "-" * 80)
            print("Verificando valores negativos en columnas numéricas:")
            print("-" * 80)
        
        numeric_cols = ['Reviews', 'Size', 'Price', 'Installs Numeric']
        for col in numeric_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    df = df[~negative_mask].copy()
                    self.removed_counts[col] = negative_count
                    if self.verbose:
                        print(f"✓ {col}: {negative_count} valores negativos eliminados")
                else:
                    if self.verbose:
                        print(f"✓ {col}: Sin valores negativos")
        
        if self.verbose:
            total_removed = initial_len - len(df)
            print(f"\nTotal de registros eliminados en este paso: {total_removed}")
            print(f"Registros restantes: {len(df):,}")
        
        return df


# ============================================================================
# TRANSFORMADOR 4: VALIDACIÓN DE CONSISTENCIA
# ============================================================================

class ConsistencyValidator(BaseEstimator, TransformerMixin):
    """
    Valida y corrige la consistencia entre variables:
    - Type='Free' debe tener Price=0
    - Type='Paid' debe tener Price>0
    - Infiere Type desde Price cuando Type es NaN
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.corrections = {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 3: VALIDACIÓN DE CONSISTENCIA ENTRE VARIABLES".center(80))
            print("=" * 80)
            print("\nValidando consistencia entre Type y Price:")
            print("-" * 80)
        
        # Casos inconsistentes: Type='Free' pero Price > 0
        free_but_paid = (df['Type'] == 'Free') & (df['Price'] > 0)
        free_but_paid_count = free_but_paid.sum()
        
        if self.verbose:
            print(f"\nApps marcadas como 'Free' pero con Price > 0: {free_but_paid_count}")
        
        if free_but_paid_count > 0:
            df.loc[free_but_paid, 'Type'] = 'Paid'
            self.corrections['free_to_paid'] = free_but_paid_count
            if self.verbose:
                print(f"✓ Corregido: {free_but_paid_count} apps cambiadas de 'Free' a 'Paid'")
        
        # Casos inconsistentes: Type='Paid' pero Price = 0
        paid_but_free = (df['Type'] == 'Paid') & (df['Price'] == 0)
        paid_but_free_count = paid_but_free.sum()
        
        if self.verbose:
            print(f"\nApps marcadas como 'Paid' pero con Price = 0: {paid_but_free_count}")
        
        if paid_but_free_count > 0:
            df.loc[paid_but_free, 'Type'] = 'Free'
            self.corrections['paid_to_free'] = paid_but_free_count
            if self.verbose:
                print(f"✓ Corregido: {paid_but_free_count} apps cambiadas de 'Paid' a 'Free'")
        
        # Inferir Type desde Price cuando Type es NaN
        type_missing = df['Type'].isnull()
        type_missing_count = type_missing.sum()
        
        if type_missing_count > 0:
            if self.verbose:
                print(f"\nType faltante en {type_missing_count} registros")
                print("Infiriendo Type desde Price...")
            
            df.loc[type_missing & (df['Price'] == 0), 'Type'] = 'Free'
            df.loc[type_missing & (df['Price'] > 0), 'Type'] = 'Paid'
            
            remaining_missing = df['Type'].isnull().sum()
            self.corrections['type_inferred'] = type_missing_count - remaining_missing
            
            if self.verbose:
                print(f"✓ Type inferido para {type_missing_count - remaining_missing} apps")
                print(f"Faltantes restantes: {remaining_missing}")
        
        if self.verbose:
            print("\n" + "-" * 80)
            print("Distribución final de Type:")
            print(df['Type'].value_counts())
            print("\n✓ Validación de consistencia completada")
        
        return df


# ============================================================================
# CLASE PARA DIVISIÓN ESTRATIFICADA
# ============================================================================

class StratifiedSplitter:
    """
    Realiza división estratificada del dataset en Train/Val/Test
    basándose en bins de Rating.
    """
    
    def __init__(self, test_size=0.30, val_size=0.50, random_state=42, verbose=True):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose
        self.train = None
        self.val = None
        self.test = None
    
    def split(self, df):
        """
        Divide el dataset en train/val/test de forma estratificada.
        
        Args:
            df: DataFrame limpio
            
        Returns:
            tuple: (train, val, test)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 4: DIVISIÓN ESTRATIFICADA DEL DATASET".center(80))
            print("=" * 80)
        
        # Estado antes de dividir
        if self.verbose:
            print(f"\nDataset después de limpieza básica:")
            print(f"  Total de registros: {len(df):,}")
            print(f"  Rating faltantes: {df['Rating'].isnull().sum():,} ({df['Rating'].isnull().sum()/len(df)*100:.2f}%)")
        
        # 1. ELIMINAR filas sin Rating (no podemos entrenar con ellas)
        df_model = df.dropna(subset=['Rating']).copy()
        
        if self.verbose:
            print(f"\nDespués de eliminar filas sin Rating:")
            print(f"  Registros disponibles para modelado: {len(df_model):,}")
            print(f"  Registros descartados: {len(df) - len(df_model):,}")
        
        # 2. DIVISIÓN ESTRATIFICADA: Train (70%), Temp (30%)
        if self.verbose:
            print("\n" + "-" * 80)
            print("Dividiendo dataset (estratificado por Rating)...")
            print("-" * 80)
        
        # Crear bins de Rating para estratificación más robusta
        df_model['rating_bin'] = pd.cut(
            df_model['Rating'], 
            bins=[0, 3, 4, 4.5, 5], 
            labels=['Low', 'Medium', 'High', 'VeryHigh']
        )
        
        train, temp = train_test_split(
            df_model,
            test_size=self.test_size,
            stratify=df_model['rating_bin'],
            random_state=self.random_state
        )
        
        # 3. Dividir Temp en Val (50%) y Test (50%) → 15% y 15% del total
        val, test = train_test_split(
            temp,
            test_size=self.val_size,
            stratify=temp['rating_bin'],
            random_state=self.random_state
        )
        
        # Eliminar columna auxiliar de binning
        train = train.drop(columns=['rating_bin'])
        val = val.drop(columns=['rating_bin'])
        test = test.drop(columns=['rating_bin'])
        
        # Reindexar para evitar problemas
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        
        if self.verbose:
            print(f"\n✓ División completada:")
            print(f"  Train: {len(train):,} registros ({len(train)/len(df_model)*100:.1f}%)")
            print(f"  Val:   {len(val):,} registros ({len(val)/len(df_model)*100:.1f}%)")
            print(f"  Test:  {len(test):,} registros ({len(test)/len(df_model)*100:.1f}%)")
        
        self.train = train
        self.val = val
        self.test = test
        
        return train, val, test


# ============================================================================
# IMPUTADOR SIN DATA LEAKAGE
# ============================================================================

class NoLeakageImputer:
    """
    Imputa valores faltantes SIN data leakage:
    1. Calcula estadísticas SOLO con train
    2. Aplica las mismas estadísticas a val y test
    3. Crea flags de faltantes ANTES de imputar
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        # Diccionarios para almacenar estadísticas de train
        self.size_medians = None
        self.size_global_median = None
        self.content_rating_modes = None
        self.content_rating_global_mode = None
        self.android_ver_modes = None
        self.android_ver_global_mode = None
        self.current_ver_modes = None
        self.current_ver_global_mode = None
        self.price_medians_paid = None
        self.price_global_median_paid = None
    
    def fit(self, train_df):
        """
        Calcula todas las estadísticas usando SOLO train.
        
        Args:
            train_df: DataFrame de entrenamiento
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 5: IMPUTACIÓN DE VALORES FALTANTES (Sin Data Leakage)".center(80))
            print("=" * 80)
            print("\nMETODOLOGÍA:")
            print("  1. Calcular estadísticas SOLO con train")
            print("  2. Aplicar las mismas estadísticas a val y test")
            print("  3. Crear flags de faltantes ANTES de imputar")
            print("=" * 80)
        
        # SIZE: Medianas por Category × Type
        self.size_medians = train_df.groupby(['Category', 'Type'])['Size'].median()
        self.size_global_median = train_df['Size'].median()
        if self.verbose:
            print(f"\n✓ SIZE: {len(self.size_medians)} medianas calculadas")
        
        # CONTENT RATING: Moda por Category
        self.content_rating_modes = train_df.groupby('Category')['Content Rating'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None
        )
        self.content_rating_global_mode = train_df['Content Rating'].mode()[0]
        if self.verbose:
            print(f"✓ CONTENT RATING: {len(self.content_rating_modes)} modas calculadas")
        
        # ANDROID VER: Moda por Category
        self.android_ver_modes = train_df.groupby('Category')['Android Ver'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None
        )
        self.android_ver_global_mode = train_df['Android Ver'].mode()[0]
        if self.verbose:
            print(f"✓ ANDROID VER: {len(self.android_ver_modes)} modas calculadas")
        
        # CURRENT VER: Moda por Category
        self.current_ver_modes = train_df.groupby('Category')['Current Ver'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None
        )
        self.current_ver_global_mode = train_df['Current Ver'].mode()[0]
        if self.verbose:
            print(f"✓ CURRENT VER: {len(self.current_ver_modes)} modas calculadas")
        
        # PRICE: Medianas para apps Paid por Category
        self.price_medians_paid = train_df[train_df['Type'] == 'Paid'].groupby('Category')['Price'].median()
        self.price_global_median_paid = train_df[train_df['Type'] == 'Paid']['Price'].median()
        if self.verbose:
            print(f"✓ PRICE: {len(self.price_medians_paid)} medianas calculadas para Paid apps")
        
        return self
    
    def transform(self, train_df, val_df, test_df):
        """
        Aplica imputación a train, val y test usando estadísticas de train.
        
        Args:
            train_df, val_df, test_df: DataFrames a imputar
            
        Returns:
            tuple: (train, val, test) imputados
        """
        # Crear copias
        train = train_df.copy()
        val = val_df.copy()
        test = test_df.copy()
        
        # ==============================================================================
        # CREAR FLAGS DE FALTANTES (antes de imputar)
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Creando flags de valores faltantes...")
            print("-" * 80)
        
        for df in [train, val, test]:
            df['size_missing'] = df['Size'].isnull().astype(int)
            df['content_rating_missing'] = df['Content Rating'].isnull().astype(int)
            df['android_ver_missing'] = df['Android Ver'].isnull().astype(int)
            df['current_ver_missing'] = df['Current Ver'].isnull().astype(int)
            df['price_missing'] = df['Price'].isnull().astype(int)
        
        if self.verbose:
            print("✓ Flags creados en train, val y test")
        
        # ==============================================================================
        # 1. SIZE: Imputar con medianas precalculadas
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Imputando SIZE (mediana por Category × Type)")
            print("-" * 80)
        
        for df in [train, val, test]:
            df = self._impute_size(df)
        
        if self.verbose:
            print(f"✓ Size imputado")
            print(f"   Train faltantes: {train['Size'].isnull().sum()}")
            print(f"   Val faltantes: {val['Size'].isnull().sum()}")
            print(f"   Test faltantes: {test['Size'].isnull().sum()}")
        
        # ==============================================================================
        # 2. CONTENT RATING: Imputar con modas precalculadas
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Imputando CONTENT RATING (moda por Category)")
            print("-" * 80)
        
        train = self._impute_content_rating(train)
        val = self._impute_content_rating(val)
        test = self._impute_content_rating(test)
        
        if self.verbose:
            print(f"✓ Content Rating imputado")
        
        # ==============================================================================
        # 3. ANDROID VER: Imputar con modas precalculadas
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Imputando ANDROID VER (moda por Category)")
            print("-" * 80)
        
        train = self._impute_android_ver(train)
        val = self._impute_android_ver(val)
        test = self._impute_android_ver(test)
        
        if self.verbose:
            print(f"✓ Android Ver imputado")
        
        # ==============================================================================
        # 4. CURRENT VER: Imputar con modas precalculadas
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Imputando CURRENT VER (moda por Category)")
            print("-" * 80)
        
        train = self._impute_current_ver(train)
        val = self._impute_current_ver(val)
        test = self._impute_current_ver(test)
        
        if self.verbose:
            print(f"✓ Current Ver imputado")
        
        # ==============================================================================
        # 5. PRICE: 0 si Free, mediana por Category si Paid
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Imputando PRICE (0 si Free, mediana por Category si Paid)")
            print("-" * 80)
        
        train = self._impute_price(train)
        val = self._impute_price(val)
        test = self._impute_price(test)
        
        if self.verbose:
            print(f"✓ Price imputado")
        
        # ==============================================================================
        # RESUMEN FINAL
        # ==============================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("IMPUTACIÓN COMPLETADA SIN DATA LEAKAGE")
            print("=" * 80)
            print("\nValores faltantes restantes:")
            
            for name, df in [('Train', train), ('Val', val), ('Test', test)]:
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) == 0:
                    print(f"  {name}: Sin valores faltantes")
                else:
                    print(f"  {name}: {missing.to_dict()}")
            
            print("\nFlags de trazabilidad creados:")
            print("  - size_missing")
            print("  - content_rating_missing")
            print("  - android_ver_missing")
            print("  - current_ver_missing")
            print("  - price_missing")
        
        return train, val, test
    
    def fit_transform(self, train_df, val_df, test_df):
        """Fit y transform en un solo paso"""
        self.fit(train_df)
        return self.transform(train_df, val_df, test_df)
    
    # Métodos auxiliares de imputación
    def _impute_size(self, df):
        """Imputa Size usando medianas precalculadas"""
        for idx, row in df[df['Size'].isnull()].iterrows():
            cat, typ = row['Category'], row['Type']
            if (cat, typ) in self.size_medians.index:
                df.loc[idx, 'Size'] = self.size_medians.loc[(cat, typ)]
            else:
                df.loc[idx, 'Size'] = self.size_global_median
        return df
    
    def _impute_content_rating(self, df):
        """Imputa Content Rating usando modas precalculadas"""
        for idx, row in df[df['Content Rating'].isnull()].iterrows():
            cat = row['Category']
            if cat in self.content_rating_modes.index and pd.notna(self.content_rating_modes.loc[cat]):
                df.loc[idx, 'Content Rating'] = self.content_rating_modes.loc[cat]
            else:
                df.loc[idx, 'Content Rating'] = self.content_rating_global_mode
        return df
    
    def _impute_android_ver(self, df):
        """Imputa Android Ver usando modas precalculadas"""
        for idx, row in df[df['Android Ver'].isnull()].iterrows():
            cat = row['Category']
            if cat in self.android_ver_modes.index and pd.notna(self.android_ver_modes.loc[cat]):
                df.loc[idx, 'Android Ver'] = self.android_ver_modes.loc[cat]
            else:
                df.loc[idx, 'Android Ver'] = self.android_ver_global_mode
        return df
    
    def _impute_current_ver(self, df):
        """Imputa Current Ver usando modas precalculadas"""
        for idx, row in df[df['Current Ver'].isnull()].iterrows():
            cat = row['Category']
            if cat in self.current_ver_modes.index and pd.notna(self.current_ver_modes.loc[cat]):
                df.loc[idx, 'Current Ver'] = self.current_ver_modes.loc[cat]
            else:
                df.loc[idx, 'Current Ver'] = self.current_ver_global_mode
        return df
    
    def _impute_price(self, df):
        """Imputa Price (0 si Free, mediana si Paid)"""
        # Free apps → 0
        mask_free = (df['Type'] == 'Free') & df['Price'].isnull()
        df.loc[mask_free, 'Price'] = 0
        
        # Paid apps → mediana por Category
        for idx, row in df[(df['Type'] == 'Paid') & df['Price'].isnull()].iterrows():
            cat = row['Category']
            if cat in self.price_medians_paid.index and pd.notna(self.price_medians_paid.loc[cat]):
                df.loc[idx, 'Price'] = self.price_medians_paid.loc[cat]
            else:
                df.loc[idx, 'Price'] = self.price_global_median_paid
        return df

# ============================================================================
# TRANSFORMADOR 6: TRANSFORMACIONES DE VARIABLES NUMÉRICAS
# ============================================================================

class NumericTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica transformaciones logarítmicas y crea variables binarias.
    - Log-transformaciones: Reviews, Installs Numeric, Size
    - Variables binarias basadas en reglas fijas
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 6: TRANSFORMACIONES DE VARIABLES NUMÉRICAS".center(80))
            print("=" * 80)
        
        # ==============================================================================
        # 1. LOG-TRANSFORMACIONES
        # ==============================================================================
        if self.verbose:
            print("\nAplicando transformaciones logarítmicas...")
            print("-" * 80)
        
        df['Reviews_log'] = np.log1p(df['Reviews'])
        df['Installs_log'] = np.log1p(df['Installs Numeric'])
        df['Size_log'] = np.log1p(df['Size'])
        
        if self.verbose:
            print(f"✓ Reviews_log: Media {df['Reviews_log'].mean():.2f}, Mediana {df['Reviews_log'].median():.2f}")
            print(f"✓ Installs_log: Media {df['Installs_log'].mean():.2f}, Mediana {df['Installs_log'].median():.2f}")
            print(f"✓ Size_log: Media {df['Size_log'].mean():.2f}, Mediana {df['Size_log'].median():.2f}")
        
        # ==============================================================================
        # 2. VARIABLES BINARIAS
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Creando variables binarias...")
            print("-" * 80)
        
        df['is_free'] = (df['Type'] == 'Free').astype(int)
        df['is_large_app'] = (df['Size'] > 50).astype(int)
        df['has_high_installs'] = (df['Installs Numeric'] > 1000000).astype(int)
        df['is_top_category'] = df['Category'].isin(['FAMILY', 'GAME']).astype(int)
        df['is_everyone_rated'] = (df['Content Rating'] == 'Everyone').astype(int)
        df['large_and_popular'] = (df['is_large_app'] & df['has_high_installs']).astype(int)
        
        if self.verbose:
            print(f"✓ is_free: {df['is_free'].sum()} ({df['is_free'].mean()*100:.1f}%)")
            print(f"✓ is_large_app: {df['is_large_app'].sum()} ({df['is_large_app'].mean()*100:.1f}%)")
            print(f"✓ has_high_installs: {df['has_high_installs'].sum()} ({df['has_high_installs'].mean()*100:.1f}%)")
            print(f"✓ is_top_category: {df['is_top_category'].sum()} ({df['is_top_category'].mean()*100:.1f}%)")
            print(f"✓ is_everyone_rated: {df['is_everyone_rated'].sum()} ({df['is_everyone_rated'].mean()*100:.1f}%)")
            print(f"✓ large_and_popular: {df['large_and_popular'].sum()} ({df['large_and_popular'].mean()*100:.1f}%)")
        
        if self.verbose:
            print("\n✓ Transformaciones numéricas completadas")
        
        return df


# ============================================================================
# TRANSFORMADOR 7: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Crea variables derivadas sin data leakage:
    - Ratios y métricas derivadas
    - Fecha de actualización y recency
    - Popularity score normalizado con estadísticas de train
    """
    
    def __init__(self, reference_date='2025-10-02', verbose=True):
        self.reference_date = pd.to_datetime(reference_date)
        self.verbose = verbose
        self.installs_min = None
        self.installs_max = None
        self.reviews_min = None
        self.reviews_max = None
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Calcula estadísticas solo de train para normalización"""
        self.installs_min = X['Installs Numeric'].min()
        self.installs_max = X['Installs Numeric'].max()
        self.reviews_min = X['Reviews'].min()
        self.reviews_max = X['Reviews'].max()
        
        # Calcular quantiles para popularity_tier (evitar data leakage)
        self.installs_quantiles = [0] + X['Installs Numeric'].quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        
        # Calcular quantiles para days_since_update (evitar data leakage)
        # Primero procesar fechas para obtener days_since_update
        temp_df = X.copy()
        temp_df['Last Updated Parsed'] = temp_df['Last Updated'].apply(self._parse_date_flexible)
        temp_df['days_since_update'] = (self.reference_date - temp_df['Last Updated Parsed']).dt.days
        valid_days = temp_df['days_since_update'].dropna()
        
        if len(valid_days) > 0:
            percentiles = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
            self.days_quantiles = valid_days.quantile(percentiles).tolist()
            self.days_quantiles[0] = -1  # Para incluir valores negativos
            self.days_quantiles[-1] = 10000  # Para valores muy altos
        else:
            # Fallback con bins fijos
            self.days_quantiles = [-1, 30, 90, 365, 730, 10000]
        
        if self.verbose:
            print(f"\n✓ Estadísticas de train almacenadas para normalización:")
            print(f"  Installs: [{self.installs_min:.0f}, {self.installs_max:.0f}]")
            print(f"  Reviews: [{self.reviews_min:.0f}, {self.reviews_max:.0f}]")
            print(f"  Installs quantiles: {self.installs_quantiles}")
            print(f"  Days quantiles: {self.days_quantiles}")

        self._is_fitted = True
        return self
    
    def _parse_date_flexible(self, date_str):
        """Método auxiliar para parsear fechas con múltiples formatos"""
        if pd.isna(date_str):
            return pd.NaT
        
        # Intentar diferentes formatos comunes
        date_formats = [
            '%B %d, %Y',      # "January 15, 2018"
            '%b %d, %Y',      # "Jan 15, 2018" 
            '%d-%b-%y',       # "15-Jan-18"
            '%Y-%m-%d',       # "2018-01-15"
            '%m/%d/%Y',       # "01/15/2018"
            '%d/%m/%Y'        # "15/01/2018"
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Si ningún formato funciona, usar pd.to_datetime con inferencia
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except:
            return pd.NaT
    
    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() antes de transform()")

        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 7: FEATURE ENGINEERING".center(80))
            print("=" * 80)
        
        # ==============================================================================
        # 1. RATIOS Y MÉTRICAS DERIVADAS MEJORADAS
        # ==============================================================================
        if self.verbose:
            print("\nCreando ratios y métricas derivadas...")
            print("-" * 80)
        
        # Ratios básicos
        df['review_rate'] = df['Reviews'] / (df['Installs Numeric'] + 1)
        df['size_per_install'] = df['Size'] / (df['Installs Numeric'] + 1)
        
        # CORREGIDO: usar max de train para evitar data leakage
        df['market_penetration'] = df['Installs Numeric'] / self.installs_max

        
        # Métricas de competitividad
        df['price_competitiveness'] = 1 / (df['Price'] + 0.01)  # Inverso del precio
        
        # Features de temporalidad mejoradas (se crearán después del procesamiento de fechas)
        
        # Features de categorización avanzada
        df['is_premium_category'] = df['Category'].isin(['FINANCE', 'PRODUCTIVITY', 'BUSINESS']).astype(int)
        df['is_entertainment_category'] = df['Category'].isin(['GAME', 'ENTERTAINMENT', 'SOCIAL']).astype(int)
        df['is_utility_category'] = df['Category'].isin(['TOOLS', 'COMMUNICATION', 'PHOTOGRAPHY']).astype(int)
        
        # Features de tamaño relativo
        df['size_category'] = pd.cut(df['Size'], 
                                   bins=[0, 10, 50, 100, 1000, float('inf')], 
                                   labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge'])
        
        # Features de popularidad relativa - CORREGIDO: usar estadísticas de train
        df['popularity_tier'] = pd.cut(
            df['Installs Numeric'], 
            bins=self.installs_quantiles, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        if self.verbose:
            print(f"✓ review_rate: Media {df['review_rate'].mean():.6f}")
            print(f"✓ size_per_install: Media {df['size_per_install'].mean():.6f}")
        
        # ==============================================================================
        # 2. FECHA DE ACTUALIZACIÓN
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Procesando fecha de actualización...")
            print("-" * 80)
        
        df['Last Updated Parsed'] = df['Last Updated'].apply(self._parse_date_flexible)
        df['days_since_update'] = (self.reference_date - df['Last Updated Parsed']).dt.days
        
        # CORREGIDO: Usar bins predefinidos para evitar data leakage
        # No usar estadísticas del dataset actual
        df['update_recency'] = pd.cut(
            df['days_since_update'],
            bins=self.days_quantiles,
            labels=['Very Recent', 'Recent', 'Moderate', 'Old', 'Very Old', 'Ancient'],
            include_lowest=True
        ) 
        
        # Añadir features de temporalidad después del procesamiento de fechas
        df['is_recently_updated'] = (df['days_since_update'] <= 90).astype(int)
        df['update_frequency_score'] = 1 / (df['days_since_update'] + 1)  # Más reciente = mayor score
        
        if self.verbose:
            print(f"✓ days_since_update: Media {df['days_since_update'].mean():.0f} días")
            print(f"✓ update_recency creada (6 categorías)")
            print(f"✓ is_recently_updated: {df['is_recently_updated'].sum()} apps recientes")
            print(f"✓ update_frequency_score: Media {df['update_frequency_score'].mean():.6f}")
        
        # ==============================================================================
        # 3. POPULARITY SCORE (normalizado con estadísticas de train)
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Calculando popularity score...")
            print("-" * 80)
        
        installs_norm = (df['Installs Numeric'] - self.installs_min) / (self.installs_max - self.installs_min)
        reviews_norm = (df['Reviews'] - self.reviews_min) / (self.reviews_max - self.reviews_min)
        df['popularity_score'] = (installs_norm * 0.7 + reviews_norm * 0.3) * 100
        
        if self.verbose:
            print(f"✓ popularity_score: Media {df['popularity_score'].mean():.2f}")
        
        # ==============================================================================
        # 4. ELIMINAR COLUMNAS REDUNDANTES
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Eliminando columnas redundantes...")
            print("-" * 80)
        
        redundant_cols = [
            'Reviews', 'Installs Numeric', 'Size', 'Installs', 'Genres', 
            'Last Updated', 'Last Updated Parsed', 'Current Ver', 'Android Ver', 
            'days_since_update'
        ]
        
        cols_to_drop = [col for col in redundant_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        if self.verbose:
            print(f"✓ {len(cols_to_drop)} columnas eliminadas: {cols_to_drop}")
        
        # ==============================================================================
        # 5. ELIMINAR DUPLICADOS
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Eliminando duplicados...")
            print("-" * 80)
        
        initial_len = len(df)
        df = df.drop_duplicates(ignore_index=True)
        
        if self.verbose:
            print(f"✓ Duplicados eliminados: {initial_len - len(df)}")
            print(f"  Registros finales: {len(df):,}")
        
        if self.verbose:
            print("\n✓ Feature engineering completado")
        
        return df


# ============================================================================
# TRANSFORMADOR 8: MANEJO DE CATEGORÍAS RARAS
# ============================================================================

class RareCategoryHandler(BaseEstimator, TransformerMixin):
    """
    Maneja categorías raras en variables categóricas:
    - Agrupa categorías con frecuencia < threshold en 'Other'
    - Elimina categorías específicas de Content Rating
    """
    
    def __init__(self, category_threshold=70, verbose=True):
        self.category_threshold = category_threshold
        self.verbose = verbose
        self.main_categories = None
    
    def fit(self, X, y=None):
        """Identifica categorías principales en train"""
        freq_train = X['Category'].value_counts()
        self.main_categories = set(freq_train[freq_train >= self.category_threshold].index.tolist())
        
        if self.verbose:
            print(f"\n✓ Categorías principales identificadas: {len(self.main_categories)}")
            print(f"  Threshold: {self.category_threshold} ocurrencias")
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 8: MANEJO DE CATEGORÍAS RARAS".center(80))
            print("=" * 80)
        
        # ==============================================================================
        # 1. CATEGORY: Agrupar raras en 'Other'
        # ==============================================================================
        if self.verbose:
            print("\nAgrupando categorías raras en 'Other'...")
            print("-" * 80)
        
        initial_unique = df['Category'].nunique()
        df['Category'] = df['Category'].apply(lambda x: x if x in self.main_categories else 'Other')
        final_unique = df['Category'].nunique()
        
        if self.verbose:
            other_count = (df['Category'] == 'Other').sum()
            other_pct = other_count / len(df) * 100
            print(f"✓ Categorías únicas: {initial_unique} → {final_unique}")
            print(f"✓ Registros en 'Other': {other_count} ({other_pct:.1f}%)")
        
        # ==============================================================================
        # 2. CONTENT RATING: Eliminar categorías problemáticas
        # ==============================================================================
        if self.verbose:
            print("\n" + "-" * 80)
            print("Eliminando categorías problemáticas de Content Rating...")
            print("-" * 80)
        
        initial_len = len(df)
        df = df[~df['Content Rating'].isin(['Adults only 18+', 'Unrated'])].copy()
        df = df.reset_index(drop=True)
        
        if self.verbose:
            print(f"✓ Registros eliminados: {initial_len - len(df)}")
            print(f"✓ Registros finales: {len(df):,}")
            print(f"\nDistribución de Content Rating:")
            print(df['Content Rating'].value_counts())
        
        if self.verbose:
            print("\n✓ Manejo de categorías raras completado")
        
        return df
    
# ============================================================================
# ANALIZADOR DE RELEVANCIA (INFORMACIÓN MUTUA)
# ============================================================================

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Selecciona variables basándose en información mutua con el target.
    - Calcula MI scores solo con train
    - Elimina variables con MI < threshold
    - Genera visualización de importancia
    """
    
    def __init__(self, target='Rating', mi_threshold=0.005, random_state=42, verbose=True, plot=False):
        self.target = target
        self.mi_threshold = mi_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.plot = plot
        self.selected_vars = None
        self.mi_scores = None
        self.low_mi_vars = None
    
    def fit(self, X, y=None):
        """Calcula información mutua solo con train"""
        from sklearn.feature_selection import mutual_info_regression
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 9: ANÁLISIS DE RELEVANCIA (INFORMACIÓN MUTUA)".center(80))
            print("=" * 80)
        
        # Seleccionar solo columnas numéricas (excluyendo target)
        ignore_cols = [self.target]
        X_cols = [
            col for col in X.columns
            if col not in ignore_cols
            and pd.api.types.is_numeric_dtype(X[col])
        ]
        
        if self.verbose:
            print(f"\nVariables numéricas analizadas: {len(X_cols)}")
        
        # Preparar datos
        X_numeric = X[X_cols].copy()
        y_target = X[self.target]
        
        # Calcular información mutua
        mi_scores = mutual_info_regression(X_numeric, y_target, random_state=self.random_state)
        self.mi_scores = pd.DataFrame({
            'Variable': X_cols, 
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        # Identificar variables con baja MI
        self.low_mi_vars = self.mi_scores[self.mi_scores['MI_Score'] < self.mi_threshold]['Variable'].tolist()
        self.selected_vars = self.mi_scores[self.mi_scores['MI_Score'] >= self.mi_threshold]['Variable'].tolist()
        
        if self.verbose:
            print(f"\n✓ Información mutua calculada")
            print(f"\nTop 10 variables más relevantes:")
            print(self.mi_scores.head(10).to_string(index=False))
            
            print(f"\n\nVariables con baja información mutua (<{self.mi_threshold}):")
            for v in self.low_mi_vars:
                mi_val = self.mi_scores[self.mi_scores['Variable'] == v]['MI_Score'].values[0]
                print(f"  - {v}: {mi_val:.4f}")
            
            print(f"\n✓ Variables seleccionadas: {len(self.selected_vars)}")
            print(f"✓ Variables eliminadas: {len(self.low_mi_vars)}")
        
        # Visualización
        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.barh(self.mi_scores['Variable'], self.mi_scores['MI_Score'], color='teal')
            plt.xlabel('Información Mutua con Rating')
            plt.title('Importancia de Variables (Información Mutua)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        return self
    
    def transform(self, X):
        """Elimina variables con baja información mutua"""
        df = X.copy()
        
        # Eliminar variables con MI baja
        cols_to_drop = [col for col in self.low_mi_vars if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
            if self.verbose:
                print(f"\n✓ Variables eliminadas por baja MI: {len(cols_to_drop)}")
        
        return df


# ============================================================================
# ANALIZADOR DE MULTICOLINEALIDAD
# ============================================================================

class MulticollinearityRemover(BaseEstimator, TransformerMixin):
    """
    Detecta y elimina variables con alta multicolinealidad.
    - Calcula matriz de correlación con train
    - Identifica pares con |corr| > threshold
    - Elimina variables predefinidas por análisis previo
    """
    
    def __init__(self, corr_threshold=0.8, vars_to_remove=None, verbose=True, plot=False):
        self.corr_threshold = corr_threshold
        self.vars_to_remove = vars_to_remove or ['Installs_log']  # Default basado en análisis
        self.verbose = verbose
        self.plot = plot
        self.corr_matrix = None
        self.high_corr_pairs = []
    
    def fit(self, X, y=None):
        """Analiza multicolinealidad solo con train"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 10: ANÁLISIS DE MULTICOLINEALIDAD".center(80))
            print("=" * 80)
        
        # Seleccionar solo variables numéricas
        numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        
        # CORREGIDO: Calcular matriz de correlación SOLO con train
        self.corr_matrix = X[numeric_cols].corr(method='pearson')
        
        # Identificar pares con alta correlación
        if self.verbose:
            print(f"\nPares de variables con alta correlación (|corr| > {self.corr_threshold}):")
        
        high_corr = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        for col in high_corr.columns:
            for idx in high_corr.index:
                corr_val = high_corr.loc[idx, col]
                if abs(corr_val) > self.corr_threshold:
                    self.high_corr_pairs.append((idx, col, corr_val))
                    if self.verbose:
                        print(f"  • {idx} y {col}: {corr_val:+.2f}")
        
        if not self.high_corr_pairs and self.verbose:
            print("  No se encontraron pares con alta correlación")
        
        # Visualización
        if self.plot:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1)
            plt.title('Matriz de Correlación (Pearson) - Multicolinealidad')
            plt.tight_layout()
            plt.show()
        
        if self.verbose:
            print(f"\n✓ Variables a eliminar por multicolinealidad: {self.vars_to_remove}")
        
        return self
    
    def transform(self, X):
        """Elimina variables con alta multicolinealidad"""
        df = X.copy()
        
        # Eliminar variables identificadas
        cols_to_drop = [col for col in self.vars_to_remove if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
            if self.verbose:
                print(f"\n✓ Variables eliminadas: {cols_to_drop}")
                print(f"✓ Columnas restantes: {len(df.columns)}")
        
        return df






# ============================================================================
# CODIFICADOR ONE-HOT ENCODING
# ============================================================================

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica one-hot encoding a variables categóricas sin data leakage.
    - Identifica columnas categóricas en train
    - Crea dummies y almacena columnas resultantes
    - Alinea val/test con las mismas columnas de train
    """
    
    def __init__(self, exclude_cols=None, drop_first=False, verbose=True):
        self.exclude_cols = exclude_cols or ['App']
        self.drop_first = drop_first
        self.verbose = verbose
        self.cat_cols = None
        self.dummy_cols = None
        self.final_cols = None
    
    def fit(self, X, y=None):
        """Identifica columnas categóricas y estructura de dummies en train"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PASO 11: ONE-HOT ENCODING".center(80))
            print("=" * 80)
        
        # Identificar columnas categóricas
        self.cat_cols = [
            c for c in X.columns
            if (X[c].dtype == 'object' or str(X[c].dtype) == 'category')
            and c not in self.exclude_cols
        ]
        
        if self.verbose:
            print(f"\nColumnas categóricas identificadas: {len(self.cat_cols)}")
            for col in self.cat_cols:
                print(f"  - {col}: {X[col].nunique()} categorías únicas")
        
        # Crear dummies en train
        train_dummies = pd.get_dummies(X, columns=self.cat_cols, drop_first=self.drop_first)
        
        # Identificar nuevas columnas dummy
        self.dummy_cols = [col for col in train_dummies.columns 
                          if col not in X.columns or col in self.cat_cols]
        
        # Almacenar columnas finales
        self.final_cols = train_dummies.columns.tolist()
        
        if self.verbose:
            print(f"\n✓ Columnas dummy creadas: {len(self.dummy_cols)}")
            print(f"✓ Total de columnas después de encoding: {len(self.final_cols)}")
        
        return self
    
    def transform(self, X):
        """Aplica one-hot encoding y alinea columnas con train"""
        # Crear dummies
        df_dummies = pd.get_dummies(X, columns=self.cat_cols, drop_first=self.drop_first)
        
        # Agregar columnas faltantes (con 0)
        for col in self.dummy_cols:
            if col not in df_dummies.columns:
                df_dummies[col] = 0
        
        # Eliminar columnas extra
        extra_cols = set(df_dummies.columns) - set(self.final_cols)
        if extra_cols:
            df_dummies = df_dummies.drop(columns=list(extra_cols))
        
        # Reordenar columnas para match con train
        df_dummies = df_dummies[self.final_cols]
        
        if self.verbose:
            print(f"\n✓ Encoding aplicado: {df_dummies.shape[1]} columnas")
        
        return df_dummies


# ============================================================================
# PIPELINE PRINCIPAL - ORQUESTADOR
# ============================================================================

class GooglePlayDataPreparationPipeline:
    """
    Pipeline completo para preprocesamiento de datos de Google Play Store.
    Orquesta todos los transformadores manteniendo separación train/val/test.
    """
    
    def __init__(self, 
                 test_size=0.30,
                 val_size=0.50,
                 category_threshold=70,
                 mi_threshold=0.005,
                 corr_threshold=0.8,
                 vars_to_remove=None,
                 reference_date='2025-10-02',
                 random_state=42,
                 verbose=True,
                 plot=False):
        """
        Inicializa el pipeline con todos sus parámetros.
        
        Args:
            test_size: Proporción para split inicial (default: 0.30)
            val_size: Proporción de temp para val (default: 0.50)
            category_threshold: Frecuencia mínima para categorías (default: 70)
            mi_threshold: Umbral de información mutua (default: 0.01)
            corr_threshold: Umbral de correlación (default: 0.8)
            vars_to_remove: Variables a eliminar por multicolinealidad
            reference_date: Fecha de referencia para cálculos
            random_state: Semilla aleatoria
            verbose: Mostrar mensajes detallados
            plot: Generar visualizaciones
        """
        self.test_size = test_size
        self.val_size = val_size
        self.category_threshold = category_threshold
        self.mi_threshold = mi_threshold
        self.corr_threshold = corr_threshold
        self.vars_to_remove = vars_to_remove or ['Installs_log']
        self.reference_date = reference_date
        self.random_state = random_state
        self.verbose = verbose
        self.plot = plot
        
        # Componentes del pipeline
        self.numeric_converter = None
        self.duplicate_remover = None
        self.impossible_values_remover = None
        self.consistency_validator = None
        self.splitter = None
        self.imputer = None
        self.numeric_transformer = None
        self.feature_engineer = None
        self.rare_category_handler = None
        self.mutual_info_selector = None
        self.multicollinearity_remover = None
        self.one_hot_encoder = None
        
        # Datos procesados
        self.train = None
        self.val = None
        self.test = None
        
    def fit_transform(self, df):
        """
        Ejecuta el pipeline completo sobre los datos.
        
        Args:
            df: DataFrame crudo de Google Play Store
            
        Returns:
            tuple: (train, val, test) completamente procesados
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("GOOGLE PLAY STORE - PIPELINE DE PREPROCESAMIENTO".center(80))
            print("=" * 80)
            print(f"\nDataset inicial: {df.shape[0]:,} filas, {df.shape[1]} columnas")
            print("=" * 80)
        
        # ============================================================================
        # FASE 1: LIMPIEZA BÁSICA (antes del split)
        # ============================================================================
        
        # Paso 0: Conversión de columnas numéricas
        self.numeric_converter = NumericConverter(verbose=self.verbose)
        df = self.numeric_converter.fit_transform(df)
        
        # Paso 1: Eliminación de duplicados
        self.duplicate_remover = DuplicateRemover(verbose=self.verbose)
        df = self.duplicate_remover.fit_transform(df)
        
        # Paso 2: Corrección de valores imposibles
        self.impossible_values_remover = ImpossibleValuesRemover(verbose=self.verbose)
        df = self.impossible_values_remover.fit_transform(df)
        
        # Paso 3: Validación de consistencia
        self.consistency_validator = ConsistencyValidator(verbose=self.verbose)
        df = self.consistency_validator.fit_transform(df)
        
        # ============================================================================
        # FASE 2: SPLIT ESTRATIFICADO
        # ============================================================================
        
        # Paso 4: División estratificada
        self.splitter = StratifiedSplitter(
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
            verbose=self.verbose
        )
        train, val, test = self.splitter.split(df)
        
        # ============================================================================
        # FASE 3: IMPUTACIÓN (sin data leakage)
        # ============================================================================
        
        # Paso 5: Imputación sin data leakage
        self.imputer = NoLeakageImputer(verbose=self.verbose)
        train, val, test = self.imputer.fit_transform(train, val, test)
        
        # ============================================================================
        # FASE 4: TRANSFORMACIONES (fit con train, transform a todos)
        # ============================================================================
        
        # Paso 6: Transformaciones numéricas
        self.numeric_transformer = NumericTransformer(verbose=self.verbose)
        self.numeric_transformer.fit(train)
        train = self.numeric_transformer.transform(train)
        val = self.numeric_transformer.transform(val)
        test = self.numeric_transformer.transform(test)
        
        # Paso 7: Feature Engineering
        self.feature_engineer = FeatureEngineer(
            reference_date=self.reference_date,
            verbose=self.verbose
        )
        self.feature_engineer.fit(train)
        train = self.feature_engineer.transform(train)
        val = self.feature_engineer.transform(val)
        test = self.feature_engineer.transform(test)
        
        # Paso 8: Manejo de categorías raras
        self.rare_category_handler = RareCategoryHandler(
            category_threshold=self.category_threshold,
            verbose=self.verbose
        )
        self.rare_category_handler.fit(train)
        train = self.rare_category_handler.transform(train)
        val = self.rare_category_handler.transform(val)
        test = self.rare_category_handler.transform(test)
        
        # ============================================================================
        # FASE 5: SELECCIÓN DE FEATURES
        # ============================================================================
        
        # Paso 9: Análisis de información mutua
        self.mutual_info_selector = MutualInfoSelector(
            target='Rating',
            mi_threshold=self.mi_threshold,
            random_state=self.random_state,
            verbose=self.verbose,
            plot=self.plot
        )
        self.mutual_info_selector.fit(train)
        train = self.mutual_info_selector.transform(train)
        val = self.mutual_info_selector.transform(val)
        test = self.mutual_info_selector.transform(test)
        
        # Paso 10: Análisis de multicolinealidad
        self.multicollinearity_remover = MulticollinearityRemover(
            corr_threshold=self.corr_threshold,
            vars_to_remove=self.vars_to_remove,
            verbose=self.verbose,
            plot=self.plot
        )
        self.multicollinearity_remover.fit(train)
        train = self.multicollinearity_remover.transform(train)
        val = self.multicollinearity_remover.transform(val)
        test = self.multicollinearity_remover.transform(test)


        # ============================================================================
        # RESUMEN INTERMEDIO: ANTES DEL ONE-HOT ENCODING
        # ============================================================================
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("RESUMEN INTERMEDIO: DATASET ANTES DE ONE-HOT ENCODING".center(80))
            print("=" * 80)
            
            # Comparación de dimensiones
            print("\n" + "=" * 80)
            print("DIMENSIONES DE LOS CONJUNTOS")
            print("=" * 80)
            
            summary_data = []
            for name, df in [('Train', train), ('Val', val), ('Test', test)]:
                summary_data.append({
                    'Conjunto': name,
                    'Registros': f"{len(df):,}",
                    'Columnas': len(df.columns),
                    'Duplicados': df.duplicated().sum(),
                    'Memoria (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            print("\n", summary_df.to_string(index=False))
            
            # Listado completo de features
            print("\n" + "=" * 80)
            print("LISTADO COMPLETO DE FEATURES")
            print("=" * 80)
            
            all_cols = train.columns.tolist()
            
            # Separar por tipo
            numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(train[col])]
            categorical_cols = [col for col in all_cols if train[col].dtype == 'object' or str(train[col].dtype) == 'category']
            
            print(f"\nTotal de features: {len(all_cols)}")
            print(f"  • Variables numéricas: {len(numeric_cols)}")
            print(f"  • Variables categóricas: {len(categorical_cols)}")
            
            # Numéricas
            print("\n" + "-" * 80)
            print(f"VARIABLES NUMÉRICAS ({len(numeric_cols)}):")
            print("-" * 80)
            for i, col in enumerate(numeric_cols, 1):
                dtype = train[col].dtype
                n_unique = train[col].nunique()
                missing = train[col].isnull().sum()
                print(f"  {i:3d}. {col:<30} | dtype: {dtype} | unique: {n_unique:>6} | missing: {missing}")
            
            # Categóricas
            if categorical_cols:
                print("\n" + "-" * 80)
                print(f"VARIABLES CATEGÓRICAS ({len(categorical_cols)}):")
                print("-" * 80)
                for i, col in enumerate(categorical_cols, 1):
                    n_unique = train[col].nunique()
                    missing = train[col].isnull().sum()
                    print(f"  {i:3d}. {col:<30} | unique: {n_unique:>6} | missing: {missing}")
                    if n_unique <= 10:
                        top_values = train[col].value_counts().head(5)
                        print(f"       Top 5: {dict(top_values)}")
            
            # Estadísticas descriptivas de variables numéricas clave
            print("\n" + "=" * 80)
            print("ESTADÍSTICAS DESCRIPTIVAS (Train - Variables Numéricas Clave)")
            print("=" * 80)
            
            key_numeric = [
                'Rating', 'Price', 'Reviews_log', 'Size_log',
                'review_rate', 'size_per_install', 'popularity_score'
            ]
            available_numeric = [col for col in key_numeric if col in train.columns]
            
            if available_numeric:
                stats = train[available_numeric].describe().T
                print("\n" + stats.to_string())
            
            # Verificar consistencia entre splits
            print("\n" + "=" * 80)
            print("VERIFICACIÓN DE CONSISTENCIA ENTRE SPLITS")
            print("=" * 80)
            
            # Verificar que todas tienen las mismas columnas
            train_cols = set(train.columns)
            val_cols = set(val.columns)
            test_cols = set(test.columns)
            
            if train_cols == val_cols == test_cols:
                print("✓ Todos los conjuntos tienen las mismas columnas")
            else:
                print("⚠ ADVERTENCIA: Los conjuntos tienen columnas diferentes")
                if train_cols != val_cols:
                    print(f"  Train - Val: {train_cols - val_cols} | {val_cols - train_cols}")
                if train_cols != test_cols:
                    print(f"  Train - Test: {train_cols - test_cols} | {test_cols - train_cols}")
            
            # Verificar valores faltantes
            train_missing = train.isnull().sum().sum()
            val_missing = val.isnull().sum().sum()
            test_missing = test.isnull().sum().sum()
            
            print(f"\nValores faltantes totales:")
            print(f"  Train: {train_missing} ({train_missing/train.size*100:.4f}%)")
            print(f"  Val:   {val_missing} ({val_missing/val.size*100:.4f}%)")
            print(f"  Test:  {test_missing} ({test_missing/test.size*100:.4f}%)")
            
            if train_missing == 0 and val_missing == 0 and test_missing == 0:
                print("✓ No hay valores faltantes en ningún conjunto")
            else:
                print("\n⚠ Columnas con valores faltantes:")
                for name, df in [('Train', train), ('Val', val), ('Test', test)]:
                    missing_cols = df.isnull().sum()
                    missing_cols = missing_cols[missing_cols > 0]
                    if len(missing_cols) > 0:
                        print(f"\n  {name}:")
                        for col, count in missing_cols.items():
                            print(f"    - {col}: {count} ({count/len(df)*100:.2f}%)")
            
            # Distribución del target
            if 'Rating' in train.columns:
                print("\n" + "=" * 80)
                print("DISTRIBUCIÓN DEL TARGET (Rating)")
                print("=" * 80)
                
                for name, df in [('Train', train), ('Val', val), ('Test', test)]:
                    print(f"\n{name}:")
                    print(f"  Media: {df['Rating'].mean():.4f}")
                    print(f"  Mediana: {df['Rating'].median():.4f}")
                    print(f"  Std: {df['Rating'].std():.4f}")
                    print(f"  Min: {df['Rating'].min():.4f}")
                    print(f"  Max: {df['Rating'].max():.4f}")
            
            print("\n" + "=" * 80)
            print("✓ RESUMEN COMPLETADO - LISTO PARA ONE-HOT ENCODING")
            print("=" * 80)
        
        # ============================================================================
        # FASE 6: ENCODING FINAL
        # ============================================================================
        
        # Paso 11: One-hot encoding
        self.one_hot_encoder = OneHotEncoder(
            exclude_cols=['App'],
            drop_first=False,
            verbose=self.verbose
        )
        self.one_hot_encoder.fit(train)
        train = self.one_hot_encoder.transform(train)
        val = self.one_hot_encoder.transform(val)
        test = self.one_hot_encoder.transform(test)
        
        # ============================================================================
        # RESUMEN FINAL
        # ============================================================================
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETADO".center(80))
            print("=" * 80)
            print("\nDatasets finales:")
            print(f"  Train: {train.shape[0]:,} filas × {train.shape[1]} columnas")
            print(f"  Val:   {val.shape[0]:,} filas × {val.shape[1]} columnas")
            print(f"  Test:  {test.shape[0]:,} filas × {test.shape[1]} columnas")
            
            # Verificar que todas tienen las mismas columnas
            assert list(train.columns) == list(val.columns) == list(test.columns), \
                "Error: Los conjuntos no tienen las mismas columnas"
            
            print("\n✓ Todos los conjuntos tienen las mismas columnas")
            print(f"✓ Sin data leakage: Todas las estadísticas calculadas solo con train")
            
            # Verificar valores faltantes
            train_missing = train.isnull().sum().sum()
            val_missing = val.isnull().sum().sum()
            test_missing = test.isnull().sum().sum()
            
            print(f"\nValores faltantes finales:")
            print(f"  Train: {train_missing}")
            print(f"  Val:   {val_missing}")
            print(f"  Test:  {test_missing}")
        
        # Antes de devolver, convertir columnas boolean-like a 0/1 para persistencia
        # Esto evita que valores 'True'/'False' (strings) o bool dtype se guarden/lee
        # como no numéricos y se pierdan en la selección de features.
        for df in [train, val, test]:
            # Objetos que contienen solo 'True'/'False' (case-insensitive)
            obj_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in obj_cols:
                uniques = df[col].dropna().unique()
                lower_uniques = set([str(u).lower() for u in uniques])
                if lower_uniques <= {'true', 'false'}:
                    df[col] = df[col].map(lambda x: 1 if str(x).lower() == 'true' else 0 if str(x).lower() == 'false' else np.nan)

            # Convertir bool a int
            bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
            for col in bool_cols:
                df[col] = df[col].astype(int)

            # Asegurar tipo numérico para columnas enteras con valores 0/1 almacenados como floats
            # (por ejemplo después de mapear), convertir a tipo int cuando no haya nulos
            for col in df.select_dtypes(include=['number']).columns:
                # Si los valores únicos son subset de {0,1} y no hay nulos, convertir a int8
                uniques = pd.unique(df[col].dropna())
                if set(uniques).issubset({0, 1}):
                    if df[col].isnull().sum() == 0:
                        df[col] = df[col].astype('int8')
    
        # Guardar datasets procesados en atributos
        self.train = train
        self.val = val
        self.test = test

        return train, val, test
    def get_feature_names(self):
        """Retorna los nombres de las features finales"""
        if self.train is not None:
            return self.train.columns.tolist()
        return None
    
    def get_summary(self):
        """Retorna un resumen del pipeline ejecutado"""
        summary = {
            'train_shape': self.train.shape if self.train is not None else None,
            'val_shape': self.val.shape if self.val is not None else None,
            'test_shape': self.test.shape if self.test is not None else None,
            'total_features': len(self.train.columns) if self.train is not None else None,
            'parameters': {
                'test_size': self.test_size,
                'val_size': self.val_size,
                'category_threshold': self.category_threshold,
                'mi_threshold': self.mi_threshold,
                'corr_threshold': self.corr_threshold,
                'vars_removed': self.vars_to_remove,
                'reference_date': self.reference_date,
                'random_state': self.random_state
            }
        }
        
        if hasattr(self.mutual_info_selector, 'selected_vars'):
            summary['selected_features'] = len(self.mutual_info_selector.selected_vars)
            summary['removed_features'] = len(self.mutual_info_selector.low_mi_vars)
        
        return summary
