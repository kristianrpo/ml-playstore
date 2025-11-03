"""
Classification Pipeline for Google Play Store Rating Prediction

Este módulo implementa un pipeline completo de clasificación multiclase para predecir
categorías de rating de aplicaciones en Google Play Store.

Familias de modelos incluidas:
- Árboles: RandomForest, ExtraTrees, GradientBoosting
- Lineales: LogisticRegression, SGDClassifier
- Boosting: XGBoost, CatBoost

NOTA: KNN ha sido removido del pipeline debido a overfitting severo (~27%)
causado por lazy learning y curse of dimensionality.

OPTIMIZACIÓN: CatBoost usa auto_class_weights='Balanced' para mejorar
recall de clase minoritaria (Low) maximizando F1-score balanceado.

Autor: Sistema de ML
Fecha: 2025
Versión: 1.3.0

Dependencias:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.1
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- catboost >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Modelos de clasificación
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')


# ============================================================================
# TRANSFORMADOR DE TARGET: RATING CONTINUO → CATEGORÍAS
# ============================================================================

class TargetTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma rating continuo [1.0-5.0] en categorías discretas.
    
    Estrategias de binning:
    - 'quantile': Divide en bins con igual número de muestras (recomendado)
    - 'uniform': Divide en bins de igual ancho
    - 'custom': Permite definir bins manualmente
    
    Ejemplo de uso:
        >>> transformer = TargetTransformer(n_bins=5, strategy='quantile')
        >>> y_train_cat = transformer.fit_transform(train['Rating'])
        >>> y_val_cat = transformer.transform(val['Rating'])
        >>> print(transformer.get_class_distribution())
    
    Args:
        n_bins: Número de categorías a crear (default: 5)
        strategy: Estrategia de binning ('quantile', 'uniform', 'custom')
        custom_bins: Lista de umbrales para strategy='custom'
        labels: Etiquetas personalizadas para las categorías
        verbose: Mostrar información detallada
    """
    
    def __init__(self, 
                 n_bins: int = 5,
                 strategy: str = 'quantile',
                 custom_bins: Optional[List[float]] = None,
                 labels: Optional[List[str]] = None,
                 verbose: bool = True):
        self.n_bins = n_bins
        self.strategy = strategy
        self.custom_bins = custom_bins
        self.labels = labels
        self.verbose = verbose
        
        # Atributos calculados en fit
        self.bin_edges_ = None
        self.label_mapping_ = None
        self.class_distribution_ = None
        self._is_fitted = False
    
    def fit(self, y: pd.Series) -> 'TargetTransformer':
        """
        Calcula los umbrales de binning basándose en los datos de entrenamiento.
        
        Args:
            y: Serie con ratings continuos
            
        Returns:
            self: Instancia fitted
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("TARGET TRANSFORMER: CONVERSIÓN A CATEGORÍAS".center(80))
            print("=" * 80)
            print(f"\nEstrategia: {self.strategy}")
            print(f"Número de bins: {self.n_bins}")
        
        # Validar datos
        if y.isnull().any():
            raise ValueError("Target contiene valores nulos. Eliminarlos antes de fit.")
        
        # Calcular bin edges según estrategia
        if self.strategy == 'quantile':
            # Quantiles para bins balanceados
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges_ = y.quantile(quantiles).values
            # Asegurar que min y max estén incluidos
            self.bin_edges_[0] = y.min() - 0.001
            self.bin_edges_[-1] = y.max() + 0.001
            
        elif self.strategy == 'uniform':
            # Bins de igual ancho
            self.bin_edges_ = np.linspace(y.min() - 0.001, y.max() + 0.001, self.n_bins + 1)
            
        elif self.strategy == 'custom':
            if self.custom_bins is None:
                raise ValueError("custom_bins debe proporcionarse para strategy='custom'")
            self.bin_edges_ = np.array(self.custom_bins)
            self.n_bins = len(self.bin_edges_) - 1
            
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
        
        # Crear etiquetas si no se proporcionaron
        if self.labels is None:
            if self.n_bins == 3:
                self.labels = ['Low', 'Medium', 'High']
            elif self.n_bins == 5:
                self.labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            else:
                self.labels = [f'Class_{i}' for i in range(self.n_bins)]
        
        # Crear mapeo de etiquetas
        self.label_mapping_ = {i: label for i, label in enumerate(self.labels)}
        
        if self.verbose:
            print(f"\nBin edges calculados:")
            for i in range(len(self.bin_edges_) - 1):
                print(f"  {self.labels[i]}: [{self.bin_edges_[i]:.3f}, {self.bin_edges_[i+1]:.3f})")
        
        self._is_fitted = True
        return self
    
    def transform(self, y: pd.Series) -> pd.Series:
        """
        Transforma ratings continuos a categorías usando los bins calculados en fit.
        
        Args:
            y: Serie con ratings continuos
            
        Returns:
            pd.Series: Serie con códigos numéricos (0, 1, 2, ...) como int
        """
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() antes de transform()")
        
        # Usar pd.cut para binning con labels numéricas
        y_binned = pd.cut(
            y,
            bins=self.bin_edges_,
            labels=False,  # Retorna índices numéricos (0, 1, 2, ...)
            include_lowest=True,
            duplicates='drop'
        )
        
        # Calcular distribución si es verbose (mostrar con etiquetas legibles)
        if self.verbose:
            # Mapear índices a etiquetas para mostrar
            y_labeled = y_binned.map(self.label_mapping_)
            dist = y_labeled.value_counts().sort_index()
            print(f"\nDistribución de clases:")
            for label, count in dist.items():
                pct = count / len(y) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Convertir a int para compatibilidad con XGBoost/LightGBM
        return y_binned.astype(int)
    
    def fit_transform(self, y: pd.Series) -> pd.Series:
        """Fit y transform en un solo paso"""
        self.fit(y)
        return self.transform(y)
    
    def get_bin_edges(self) -> List[float]:
        """Retorna los umbrales de los bins"""
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() primero")
        return self.bin_edges_.tolist()
    
    def get_class_distribution(self, y: pd.Series) -> pd.DataFrame:
        """
        Retorna DataFrame con distribución de clases.
        
        Args:
            y: Serie con categorías numéricas (después de transform)
            
        Returns:
            pd.DataFrame: Distribución con conteos y porcentajes
        """
        # Mapear códigos numéricos a etiquetas
        y_labeled = y.map(self.label_mapping_)
        dist = y_labeled.value_counts().sort_index()
        return pd.DataFrame({
            'Class': dist.index,
            'Count': dist.values,
            'Percentage': (dist.values / len(y) * 100).round(2)
        })
    
    def inverse_transform(self, y_cat: pd.Series) -> pd.Series:
        """
        Convierte categorías de vuelta a valores numéricos (punto medio del bin).
        
        Args:
            y_cat: Serie con categorías
            
        Returns:
            pd.Series: Serie con valores numéricos aproximados
        """
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() primero")
        
        # Mapear cada categoría al punto medio de su bin
        midpoints = {}
        for i, label in enumerate(self.labels):
            midpoint = (self.bin_edges_[i] + self.bin_edges_[i+1]) / 2
            midpoints[label] = midpoint
        
        return y_cat.map(midpoints)



# ============================================================================
# FAMILIA ÁRBOLES: RandomForest, ExtraTrees, GradientBoosting
# ============================================================================

class TreeClassifierFamily(BaseEstimator):
    """
    Familia de modelos basados en árboles de decisión para clasificación.
    
    No requiere escalado de datos. Incluye:
    - RandomForestClassifier con GridSearchCV
    - ExtraTreesClassifier (usa parámetros de RandomForest)
    - GradientBoostingClassifier con GridSearchCV
    
    Todos los modelos usan class_weight='balanced' para manejar desbalance.
    
    Ejemplo de uso:
        >>> tree_family = TreeClassifierFamily(cv_folds=5, random_state=42)
        >>> tree_family.fit(X_train, y_train, X_val, y_val)
        >>> predictions = tree_family.predict(X_test, model_name='Random Forest')
        >>> results = tree_family.get_results()
    
    Args:
        cv_folds: Número de folds para cross-validation (default: 5)
        random_state: Semilla para reproducibilidad (default: 42)
        n_jobs: Número de cores a usar (-1 usa todos) (default: -1)
        verbose: Mostrar información detallada (default: True)
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, 
                 n_jobs: int = -1, verbose: bool = True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.models = {}
        self.results = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'TreeClassifierFamily':
        """
        Entrena todos los modelos de árboles.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento (categórico)
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            
        Returns:
            self: Instancia entrenada
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA ÁRBOLES: RANDOM FOREST, EXTRA TREES, GRADIENT BOOSTING".center(80))
            print("=" * 80)
        
        # Combinar train + val para GridSearchCV con validación predefinida
        if X_val is not None and y_val is not None:
            X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
            y_combined = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0).reset_index(drop=True)
            # Crear split predefinido: -1 para train, 0 para val
            test_fold = [-1] * len(X_train) + [0] * len(X_val)
            ps = PredefinedSplit(test_fold)
            cv_strategy = ps
        else:
            # Si no hay val, usar StratifiedKFold tradicional
            X_combined = X_train
            y_combined = y_train
            cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # ====================================================================
        # RANDOM FOREST
        # ====================================================================
        if self.verbose:
            print("\n[1/3] Entrenando Random Forest...")
            print("-" * 80)
        
        param_grid_rf = {
            'n_estimators': [80, 100],  # Reducido más para evitar overfitting
            'max_depth': [6, 8, 10],  # Profundidades más conservadoras (era 8,10,12,15)
            'min_samples_split': [15, 20, 25],  # Más conservador (era 10,15,20)
            'min_samples_leaf': [6, 8, 10],  # Hojas más grandes (era 4,5,8)
            'max_features': ['sqrt'],  # Mantener sqrt
            'max_samples': [0.6, 0.7]  # Más agresivo con bagging (era 0.7,0.8)
        }
        
        if self.verbose:
            total_rf = (len(param_grid_rf['n_estimators']) * 
                       len(param_grid_rf['max_depth']) * 
                       len(param_grid_rf['min_samples_split']) *
                       len(param_grid_rf['min_samples_leaf']) *
                       len(param_grid_rf['max_features']) *
                       len(param_grid_rf['max_samples']))
            print(f"  Combinaciones: {total_rf}")
        
        start_time = time.time()
        
        # Usar class_weight más agresivo: duplicar peso de clase minoritaria
        aggressive_class_weight = {0: 2.5, 1: 1.0}  # Low=2.5x, High=1.0x
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(
                random_state=self.random_state,
                class_weight=aggressive_class_weight,
                n_jobs=self.n_jobs
            ),
            param_grid_rf,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        rf_grid.fit(X_combined, y_combined)
        rf_time = time.time() - start_time
        
        self.models['Random Forest'] = rf_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {rf_time:.2f}s")
            print(f"  Mejores parámetros: {rf_grid.best_params_}")
            print(f"  Mejor score en val: {rf_grid.best_score_:.4f}")
        
        # Evaluar en validación (ahora son métricas finales del mejor modelo)
        if X_val is not None and y_val is not None:
            y_pred_val = rf_grid.predict(X_val)
            y_pred_train = rf_grid.predict(X_train)
            y_proba_val = rf_grid.predict_proba(X_val)
            
            self.results['Random Forest'] = {
                'model': rf_grid.best_estimator_,
                'best_params': rf_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': rf_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0),
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_grid.best_estimator_.feature_importances_
                }).sort_values('importance', ascending=False)
            }
        
        # ====================================================================
        # EXTRA TREES
        # ====================================================================
        if self.verbose:
            print("\n[2/3] Entrenando Extra Trees...")
            print("-" * 80)
            print("  Usando parámetros similares a Random Forest")
        
        best_rf_params = rf_grid.best_params_
        
        start_time = time.time()
        
        # Usar class_weight más agresivo
        aggressive_class_weight = {0: 2.5, 1: 1.0}
        
        et_model = ExtraTreesClassifier(
            n_estimators=best_rf_params['n_estimators'],
            max_depth=best_rf_params['max_depth'],
            min_samples_split=best_rf_params['min_samples_split'],
            min_samples_leaf=best_rf_params['min_samples_leaf'],
            max_features=best_rf_params['max_features'],
            random_state=self.random_state,
            class_weight=aggressive_class_weight,
            n_jobs=self.n_jobs
        )
        
        et_model.fit(X_combined, y_combined)
        et_time = time.time() - start_time
        
        self.models['Extra Trees'] = et_model
        
        if self.verbose:
            print(f"  ✓ Completado en {et_time:.2f}s")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            y_pred_val = et_model.predict(X_val)
            y_pred_train = et_model.predict(X_train)
            y_proba_val = et_model.predict_proba(X_val)
            
            self.results['Extra Trees'] = {
                'model': et_model,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': et_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # ====================================================================
        # GRADIENT BOOSTING
        # ====================================================================
        if self.verbose:
            print("\n[3/3] Entrenando Gradient Boosting...")
            print("-" * 80)
        
        param_grid_gb = {
            'learning_rate': [0.03, 0.05, 0.08],  # Más conservador (era 0.05,0.08,0.1)
            'n_estimators': [80, 100, 120],  # Menos árboles para evitar overfitting
            'max_depth': [2, 3, 4],  # Árboles más shallow (era 3,4,5)
            'subsample': [0.6, 0.7],  # Submuestreo más agresivo (era 0.7,0.8)
            'min_samples_split': [20, 25],  # Más conservador (era 15,20)
            'min_samples_leaf': [10, 12],  # Hojas más grandes (era 8,10)
            'max_features': ['sqrt']  # Mantener sqrt
        }
        
        if self.verbose:
            total_gb = (len(param_grid_gb['learning_rate']) * 
                       len(param_grid_gb['n_estimators']) * 
                       len(param_grid_gb['max_depth']) *
                       len(param_grid_gb['subsample']) *
                       len(param_grid_gb['min_samples_split']) *
                       len(param_grid_gb['min_samples_leaf']) *
                       len(param_grid_gb['max_features']))
            print(f"  Combinaciones: {total_gb}")
        
        start_time = time.time()
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_state),
            param_grid_gb,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        gb_grid.fit(X_combined, y_combined)
        gb_time = time.time() - start_time
        
        self.models['Gradient Boosting'] = gb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {gb_time:.2f}s")
            print(f"  Mejores parámetros: {gb_grid.best_params_}")
            print(f"  Mejor score en val: {gb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            y_pred_val = gb_grid.predict(X_val)
            y_pred_train = gb_grid.predict(X_train)
            y_proba_val = gb_grid.predict_proba(X_val)
            
            self.results['Gradient Boosting'] = {
                'model': gb_grid.best_estimator_,
                'best_params': gb_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': gb_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA ÁRBOLES".center(80))
            print(f"{'=' * 80}")
            for model_name in ['Random Forest', 'Extra Trees', 'Gradient Boosting']:
                if model_name in self.results:
                    r = self.results[model_name]
                    print(f"\n  {model_name}:")
                    print(f"    Accuracy (val):  {r['accuracy_val']:.4f}")
                    print(f"    F1-score (val):  {r['f1_val']:.4f}")
                    print(f"    Tiempo:          {r['train_time']:.2f}s")
        
        return self
    
    def predict(self, X, model_name: str = 'Random Forest') -> np.ndarray:
        """Predice clases usando el modelo especificado"""
        return self.models[model_name].predict(X)
    
    def predict_proba(self, X, model_name: str = 'Random Forest') -> np.ndarray:
        """Predice probabilidades usando el modelo especificado"""
        return self.models[model_name].predict_proba(X)
    
    def get_results(self) -> Dict:
        """Retorna diccionario con todos los resultados"""
        return self.results
    
    def get_feature_importance(self, model_name: str = 'Random Forest', top_n: int = 20) -> pd.DataFrame:
        """Retorna feature importance del modelo especificado"""
        if model_name in self.results and 'feature_importance' in self.results[model_name]:
            return self.results[model_name]['feature_importance'].head(top_n)
        return None



# ============================================================================
# FAMILIA KNN: K-Nearest Neighbors - COMENTADO (overfitting 27% por lazy learning)
# ============================================================================

# class KNNClassifierFamily(BaseEstimator):
#     """
#     Familia K-Nearest Neighbors con escalado StandardScaler.
#     
#     KNN es sensible a la escala de las features, por lo que requiere
#     normalización. Se optimizan hiperparámetros con GridSearchCV.
#     
#     NOTA: Comentado porque presenta overfitting severo (~27%) debido a:
#     - Lazy learning: memoriza datos de entrenamiento en lugar de aprender patrones
#     - Curse of dimensionality con 66 features
#     - weights='distance' da peso infinito a matches exactos en train
#     
#     Ejemplo de uso:
#         >>> knn_family = KNNClassifierFamily(cv_folds=5, random_state=42)
#         >>> knn_family.fit(X_train, y_train, X_val, y_val)
#         >>> predictions = knn_family.predict(X_test)
#     
#     Args:
#         cv_folds: Número de folds para cross-validation (default: 5)
#         random_state: Semilla para reproducibilidad (default: 42)
#         n_jobs: Número de cores a usar (-1 usa todos) (default: -1)
#         verbose: Mostrar información detallada (default: True)
#     """
#     
#     def __init__(self, cv_folds: int = 5, random_state: int = 42,
#                  n_jobs: int = -1, verbose: bool = True):
#         self.cv_folds = cv_folds
#         self.random_state = random_state
#         self.n_jobs = n_jobs
#         self.verbose = verbose
#         
#         self.scaler = None
#         self.model = None
#         self.results = {}
#     
#     def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'KNNClassifierFamily':
#         """Entrena KNN con escalado y optimización de hiperparámetros"""
#         if self.verbose:
#             print("\n" + "=" * 80)
#             print("FAMILIA KNN: K-NEAREST NEIGHBORS".center(80))
#             print("=" * 80)
#         
#         # Escalado
#         if self.verbose:
#             print("\n[1/2] Aplicando StandardScaler...")
#         
#         self.scaler = StandardScaler()
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
#         
#         if self.verbose:
#             print(f"  ✓ Datos escalados: μ={X_train_scaled.mean():.6f}, σ={X_train_scaled.std():.6f}")
#         
#         # GridSearchCV
#         if self.verbose:
#             print("\n[2/2] Entrenando KNN con GridSearchCV...")
#             print("-" * 80)
#         
#         param_grid_knn = {
#             'n_neighbors': [40, 50, 60, 80] ,  # Aumentar vecinos para reducir overfitting (era 3,5,7,11,15)
#             'weights': ['distance'],  # Solo distance para dar más peso a vecinos cercanos
#             'metric': ['manhattan'],  # Solo manhattan (funcionó bien antes)
#             'algorithm': ['auto']  # Solo auto
#         }
#         
#         if self.verbose:
#             total_knn = (len(param_grid_knn['n_neighbors']) * 
#                         len(param_grid_knn['weights']) * 
#                         len(param_grid_knn['metric']) *
#                         len(param_grid_knn['algorithm']))
#             print(f"  Combinaciones: {total_knn}")
#         
#         start_time = time.time()
#         
#         knn_grid = GridSearchCV(
#             KNeighborsClassifier(n_jobs=self.n_jobs),
#             param_grid_knn,
#             cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
#             scoring='f1_weighted',
#             n_jobs=self.n_jobs,
#             verbose=1 if self.verbose else 0
#         )
#         
#         knn_grid.fit(X_train_scaled, y_train)
#         train_time = time.time() - start_time
#         
#         self.model = knn_grid.best_estimator_
#         
#         if self.verbose:
#             print(f"  ✓ Completado en {train_time:.2f}s")
#             print(f"  Mejores parámetros: {knn_grid.best_params_}")
#             print(f"  Mejor CV F1-score: {knn_grid.best_score_:.4f}")
#         
#         # Evaluar
#         if X_val_scaled is not None and y_val is not None:
#             # Medir tiempo de predicción
#             pred_start = time.time()
#             y_pred_val = knn_grid.predict(X_val_scaled)
#             pred_time = time.time() - pred_start
#             
#             y_pred_train = knn_grid.predict(X_train_scaled)
#             y_proba_val = knn_grid.predict_proba(X_val_scaled)
#             
#             self.results['KNN'] = {
#                 'model': knn_grid.best_estimator_,
#                 'scaler': self.scaler,
#                 'best_params': knn_grid.best_params_,
#                 'accuracy_val': accuracy_score(y_val, y_pred_val),
#                 'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
#                 'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
#                 'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
#                 'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
#                 'accuracy_train': accuracy_score(y_train, y_pred_train),
#                 'train_time': train_time,
#                 'pred_time': pred_time,
#                 'predictions_val': y_pred_val,
#                 'probabilities_val': y_proba_val,
#                 'confusion_matrix': confusion_matrix(y_val, y_pred_val),
#                 'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
#             }
#             
#             if self.verbose:
#                 print(f"\n  Métricas en validación:")
#                 print(f"    Accuracy:  {self.results['KNN']['accuracy_val']:.4f}")
#                 print(f"    F1-score:  {self.results['KNN']['f1_val']:.4f}")
#                 print(f"    Tiempo de predicción: {pred_time:.4f}s")
#                 
#                 # Advertencia si predicción es lenta
#                 if pred_time > 1.0:
#                     print(f"\n  ⚠ ADVERTENCIA: Tiempo de predicción alto ({pred_time:.2f}s)")
#                     print(f"    Considerar usar algorithm='ball_tree' o reducir n_neighbors")
#         
#         return self
#     
#     def predict(self, X) -> np.ndarray:
#         """Predice clases (escala datos automáticamente)"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict(X_scaled)
#     
#     def predict_proba(self, X) -> np.ndarray:
#         """Predice probabilidades (escala datos automáticamente)"""
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict_proba(X_scaled)
#     
#     def get_results(self) -> Dict:
#         """Retorna diccionario con todos los resultados"""
#         return self.results



# ============================================================================
# FAMILIA LINEALES: LogisticRegression, SGDClassifier
# ============================================================================

class LinearClassifierFamily(BaseEstimator):
    """
    Familia de modelos lineales regularizados para clasificación.
    
    Incluye LogisticRegression y SGDClassifier con StandardScaler.
    Ambos modelos usan class_weight='balanced'.
    
    Ejemplo de uso:
        >>> linear_family = LinearClassifierFamily(cv_folds=5, random_state=42)
        >>> linear_family.fit(X_train, y_train, X_val, y_val)
        >>> predictions = linear_family.predict(X_test, model_name='Logistic Regression')
    
    Args:
        cv_folds: Número de folds para cross-validation (default: 5)
        random_state: Semilla para reproducibilidad (default: 42)
        n_jobs: Número de cores a usar (-1 usa todos) (default: -1)
        verbose: Mostrar información detallada (default: True)
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42,
                 n_jobs: int = -1, verbose: bool = True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.scaler = None
        self.models = {}
        self.results = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'LinearClassifierFamily':
        """Entrena modelos lineales con escalado"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA LINEALES: LOGISTIC REGRESSION, SGD CLASSIFIER".center(80))
            print("=" * 80)
        
        # Combinar train y val para GridSearchCV con PredefinedSplit
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        cv_strategy = PredefinedSplit(test_fold)
        
        # Escalado
        if self.verbose:
            print("\n[0/2] Aplicando StandardScaler...")
        
        self.scaler = StandardScaler()
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        if self.verbose:
            print(f"  ✓ Datos escalados")
        
        # ====================================================================
        # LOGISTIC REGRESSION
        # ====================================================================
        if self.verbose:
            print("\n[1/2] Entrenando Logistic Regression...")
            print("-" * 80)
        
        param_grid_lr = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['saga', 'liblinear']
        }
        
        if self.verbose:
            total_lr = (len(param_grid_lr['C']) * 
                       len(param_grid_lr['penalty']) * 
                       len(param_grid_lr['solver']))
            print(f"  Combinaciones: {total_lr}")
        
        start_time = time.time()
        
        lr_grid = GridSearchCV(
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                multi_class='ovr',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            param_grid_lr,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        lr_grid.fit(X_combined_scaled, y_combined)
        lr_time = time.time() - start_time
        
        self.models['Logistic Regression'] = lr_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {lr_time:.2f}s")
            print(f"  Mejores parámetros: {lr_grid.best_params_}")
            print(f"  Mejor score en val: {lr_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_train_scaled = self.scaler.transform(X_train)
            
            y_pred_val = lr_grid.predict(X_val_scaled)
            y_pred_train = lr_grid.predict(X_train_scaled)
            y_proba_val = lr_grid.predict_proba(X_val_scaled)
            
            self.results['Logistic Regression'] = {
                'model': lr_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': lr_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': lr_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # ====================================================================
        # SGD CLASSIFIER
        # ====================================================================
        if self.verbose:
            print("\n[2/2] Entrenando SGD Classifier...")
            print("-" * 80)
        
        param_grid_sgd = {
            'alpha': [0.0001, 0.001, 0.01],
            'loss': ['hinge', 'log_loss'],
            'penalty': ['l1', 'l2', 'elasticnet']
        }
        
        if self.verbose:
            total_sgd = (len(param_grid_sgd['alpha']) * 
                        len(param_grid_sgd['loss']) * 
                        len(param_grid_sgd['penalty']))
            print(f"  Combinaciones: {total_sgd}")
        
        start_time = time.time()
        
        sgd_grid = GridSearchCV(
            SGDClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            param_grid_sgd,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        sgd_grid.fit(X_combined_scaled, y_combined)
        sgd_time = time.time() - start_time
        
        self.models['SGD Classifier'] = sgd_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {sgd_time:.2f}s")
            print(f"  Mejores parámetros: {sgd_grid.best_params_}")
            print(f"  Mejor score en val: {sgd_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_train_scaled = self.scaler.transform(X_train)
            
            y_pred_val = sgd_grid.predict(X_val_scaled)
            y_pred_train = sgd_grid.predict(X_train_scaled)
            
            self.results['SGD Classifier'] = {
                'model': sgd_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': sgd_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': sgd_time,
                'predictions_val': y_pred_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA LINEALES".center(80))
            print(f"{'=' * 80}")
            for model_name in ['Logistic Regression', 'SGD Classifier']:
                if model_name in self.results:
                    r = self.results[model_name]
                    print(f"\n  {model_name}:")
                    print(f"    Accuracy (val):  {r['accuracy_val']:.4f}")
                    print(f"    F1-score (val):  {r['f1_val']:.4f}")
                    print(f"    Tiempo:          {r['train_time']:.2f}s")
        
        return self
    
    def predict(self, X, model_name: str = 'Logistic Regression') -> np.ndarray:
        """Predice clases (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)
    
    def predict_proba(self, X, model_name: str = 'Logistic Regression') -> np.ndarray:
        """Predice probabilidades (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        # SGDClassifier no tiene predict_proba con loss='hinge'
        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(X_scaled)
        else:
            # Usar decision_function como alternativa
            return self.models[model_name].decision_function(X_scaled)
    
    def get_results(self) -> Dict:
        """Retorna diccionario con todos los resultados"""
        return self.results
    
    def get_coefficients(self, model_name: str = 'Logistic Regression') -> pd.DataFrame:
        """Retorna coeficientes del modelo para interpretabilidad"""
        if model_name in self.models and hasattr(self.models[model_name], 'coef_'):
            # Para multiclase, coef_ tiene shape (n_classes, n_features)
            # Retornar promedio absoluto de coeficientes
            coefs = np.abs(self.models[model_name].coef_).mean(axis=0)
            return pd.DataFrame({
                'feature': range(len(coefs)),
                'coefficient': coefs
            }).sort_values('coefficient', ascending=False)
        return None



# ============================================================================
# FAMILIA BOOSTING: XGBoost, LightGBM, CatBoost
# ============================================================================

class BoostingClassifierFamily(BaseEstimator):
    """
    Familia de modelos de boosting avanzados para clasificación.
    
    Incluye XGBoost, LightGBM y CatBoost con MinMaxScaler.
    Todos los modelos manejan desbalance de clases automáticamente.
    
    Ejemplo de uso:
        >>> boosting_family = BoostingClassifierFamily(cv_folds=5, random_state=42)
        >>> boosting_family.fit(X_train, y_train, X_val, y_val)
        >>> predictions = boosting_family.predict(X_test, model_name='CatBoost')
    
    Args:
        cv_folds: Número de folds para cross-validation (default: 5)
        random_state: Semilla para reproducibilidad (default: 42)
        n_jobs: Número de cores a usar (-1 usa todos) (default: -1)
        verbose: Mostrar información detallada (default: True)
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42,
                 n_jobs: int = -1, verbose: bool = True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.scaler = None
        self.models = {}
        self.results = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'BoostingClassifierFamily':
        """Entrena modelos de boosting con escalado"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA BOOSTING: XGBOOST, CATBOOST, LIGHTGBM".center(80))
            print("=" * 80)
        
        # Combinar train y val para GridSearchCV con PredefinedSplit
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        cv_strategy = PredefinedSplit(test_fold)
        
        # Escalado
        if self.verbose:
            print("\n[0/3] Aplicando MinMaxScaler...")
        
        self.scaler = MinMaxScaler()
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        if self.verbose:
            print(f"  ✓ Datos escalados: rango=[{X_combined_scaled.min():.4f}, {X_combined_scaled.max():.4f}]")
        
        # ====================================================================
        # XGBOOST
        # ====================================================================
        if self.verbose:
            print("\n[1/2] Entrenando XGBoost...")
            print("-" * 80)
        
        param_grid_xgb = {
            'n_estimators': [100, 150],  # Reducido para evitar overfitting
            'learning_rate': [0.05, 0.08],  # Learning rates más conservadores
            'max_depth': [4, 5],  # Árboles menos profundos
            'subsample': [0.7, 0.8],  # Submuestreo más agresivo
            'colsample_bytree': [0.7, 0.8],  # Feature sampling
            'reg_alpha': [1.0, 2.0],  # Regularización L1
            'reg_lambda': [2.0, 3.0],  # Regularización L2 aumentada
            'min_child_weight': [5, 8]  # Mínimo peso en nodos hoja aumentado
        }
        
        if self.verbose:
            total_xgb = (len(param_grid_xgb['n_estimators']) * 
                        len(param_grid_xgb['learning_rate']) * 
                        len(param_grid_xgb['max_depth']) *
                        len(param_grid_xgb['subsample']) *
                        len(param_grid_xgb['colsample_bytree']) *
                        len(param_grid_xgb['reg_alpha']) *
                        len(param_grid_xgb['reg_lambda']) *
                        len(param_grid_xgb['min_child_weight']))
            print(f"  Combinaciones: {total_xgb}")
        
        start_time = time.time()
        
        # Calcular scale_pos_weight para XGBoost: ratio High/Low
        # Low=0 (minoritaria), High=1 (mayoritaria)
        n_low = (y_combined == 0).sum()
        n_high = (y_combined == 1).sum()
        scale_pos_weight = n_high / n_low  # ~2.58 si Low=27.9%
        
        xgb_grid = GridSearchCV(
            XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='mlogloss',
                scale_pos_weight=scale_pos_weight  # Penaliza más errores en Low
            ),
            param_grid_xgb,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        xgb_grid.fit(X_combined_scaled, y_combined)
        xgb_time = time.time() - start_time
        
        self.models['XGBoost'] = xgb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {xgb_time:.2f}s")
            print(f"  Mejores parámetros: {xgb_grid.best_params_}")
            print(f"  Mejor score en val: {xgb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_train_scaled = self.scaler.transform(X_train)
            
            y_pred_val = xgb_grid.predict(X_val_scaled)
            y_pred_train = xgb_grid.predict(X_train_scaled)
            y_proba_val = xgb_grid.predict_proba(X_val_scaled)
            
            self.results['XGBoost'] = {
                'model': xgb_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': xgb_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': xgb_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # ====================================================================
        # CATBOOST
        # ====================================================================
        if self.verbose:
            print("\n[2/3] Entrenando CatBoost...")
            print("-" * 80)
        
        param_grid_catboost = {
            'iterations': [100, 150],  # Reducido para evitar overfitting
            'learning_rate': [0.05, 0.08],  # Learning rates conservadores
            'depth': [4, 5],  # Árboles más simples
            'l2_leaf_reg': [3, 5, 8],  # Regularización fuerte
            'subsample': [0.7, 0.8],  # Nuevo: bagging más agresivo
            'colsample_bylevel': [0.7, 0.8]  # Nuevo: feature sampling
        }
        
        if self.verbose:
            total_catboost = (len(param_grid_catboost['iterations']) * 
                             len(param_grid_catboost['learning_rate']) * 
                             len(param_grid_catboost['depth']) *
                             len(param_grid_catboost['l2_leaf_reg']) *
                             len(param_grid_catboost['subsample']) *
                             len(param_grid_catboost['colsample_bylevel']))
            print(f"  Combinaciones: {total_catboost}")
        
        start_time = time.time()
        
        # Calcular class_weights para CatBoost: Low=2.5x, High=1.0x
        catboost_class_weights = [2.5, 1.0]  # [Low, High]
        
        catboost_grid = GridSearchCV(
            CatBoostClassifier(
                random_state=self.random_state,
                verbose=0,  # Silenciar logs de CatBoost
                thread_count=self.n_jobs if self.n_jobs > 0 else -1,
                class_weights=catboost_class_weights  # Penaliza más errores en Low
            ),
            param_grid_catboost,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=1,  # CatBoost ya usa múltiples threads internamente
            verbose=1 if self.verbose else 0
        )
        
        catboost_grid.fit(X_combined_scaled, y_combined)
        catboost_time = time.time() - start_time
        
        self.models['CatBoost'] = catboost_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {catboost_time:.2f}s")
            print(f"  Mejores parámetros: {catboost_grid.best_params_}")
            print(f"  Mejor score en val: {catboost_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_train_scaled = self.scaler.transform(X_train)
            
            y_pred_val = catboost_grid.predict(X_val_scaled)
            y_pred_train = catboost_grid.predict(X_train_scaled)
            y_proba_val = catboost_grid.predict_proba(X_val_scaled)
            
            self.results['CatBoost'] = {
                'model': catboost_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': catboost_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': catboost_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # ====================================================================
        # LIGHTGBM - COMENTADO (muy lento)
        # ====================================================================
        # if self.verbose:
        #     print("\n[2/2] Entrenando LightGBM...")
        #     print("-" * 80)
        # 
        # param_grid_lgb = {
        #     'n_estimators': [100],  # Reducido de 2 a 1 opción
        #     'learning_rate': [0.05, 0.1],  # Reducido de 3 a 2 opciones
        #     'max_depth': [5],  # Reducido de 3 a 1 opción
        #     'subsample': [0.8],  # Reducido de 2 a 1 opción
        #     'colsample_bytree': [0.8],  # Reducido de 2 a 1 opción
        #     'reg_alpha': [0.5],  # Reducido de 2 a 1 opción
        #     'reg_lambda': [1.0],  # Reducido de 2 a 1 opción
        #     'min_child_samples': [10]  # Reducido de 2 a 1 opción
        # }
        # 
        # if self.verbose:
        #     total_lgb = (len(param_grid_lgb['n_estimators']) * 
        #                 len(param_grid_lgb['learning_rate']) * 
        #                 len(param_grid_lgb['max_depth']) *
        #                 len(param_grid_lgb['subsample']) *
        #                 len(param_grid_lgb['colsample_bytree']) *
        #                 len(param_grid_lgb['reg_alpha']) *
        #                 len(param_grid_lgb['reg_lambda']) *
        #                 len(param_grid_lgb['min_child_samples']))
        #     print(f"  Combinaciones: {total_lgb}")
        # 
        # start_time = time.time()
        # 
        # lgb_grid = GridSearchCV(
        #     LGBMClassifier(
        #         random_state=self.random_state,
        #         n_jobs=self.n_jobs,
        #         class_weight='balanced',
        #         verbose=-1
        #     ),
        #     param_grid_lgb,
        #     cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
        #     scoring='f1_weighted',
        #     n_jobs=self.n_jobs,
        #     verbose=1 if self.verbose else 0
        # )
        # 
        # lgb_grid.fit(X_train_scaled, y_train)
        # lgb_time = time.time() - start_time
        # 
        # self.models['LightGBM'] = lgb_grid.best_estimator_
        # 
        # if self.verbose:
        #     print(f"  ✓ Completado en {lgb_time:.2f}s")
        #     print(f"  Mejores parámetros: {lgb_grid.best_params_}")
        #     print(f"  Mejor CV F1-score: {lgb_grid.best_score_:.4f}")
        # 
        # # Evaluar
        # if X_val_scaled is not None and y_val is not None:
        #     y_pred_val = lgb_grid.predict(X_val_scaled)
        #     y_pred_train = lgb_grid.predict(X_train_scaled)
        #     y_proba_val = lgb_grid.predict_proba(X_val_scaled)
        #     
        #     self.results['LightGBM'] = {
        #         'model': lgb_grid.best_estimator_,
        #         'scaler': self.scaler,
        #         'best_params': lgb_grid.best_params_,
        #         'accuracy_val': accuracy_score(y_val, y_pred_val),
        #         'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
        #         'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
        #         'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
        #         'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
        #         'accuracy_train': accuracy_score(y_train, y_pred_train),
        #         'train_time': lgb_time,
        #         'predictions_val': y_pred_val,
        #         'probabilities_val': y_proba_val,
        #         'confusion_matrix': confusion_matrix(y_val, y_pred_val),
        #         'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
        #     }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA BOOSTING".center(80))
            print(f"{'=' * 80}")
            for model_name in ['XGBoost', 'CatBoost']:
                if model_name in self.results:
                    r = self.results[model_name]
                    print(f"\n  {model_name}:")
                    print(f"    Accuracy (val):  {r['accuracy_val']:.4f}")
                    print(f"    F1-score (val):  {r['f1_val']:.4f}")
                    print(f"    Tiempo:          {r['train_time']:.2f}s")
        
        return self
    
    def predict(self, X, model_name: str = 'LightGBM') -> np.ndarray:
        """Predice clases (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)
    
    def predict_proba(self, X, model_name: str = 'LightGBM') -> np.ndarray:
        """Predice probabilidades (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict_proba(X_scaled)
    
    def get_results(self) -> Dict:
        """Retorna diccionario con todos los resultados"""
        return self.results




# ============================================================================
# ORQUESTADOR PRINCIPAL - CLASSIFICATION TRAINING PIPELINE
# ============================================================================

class ClassificationTrainingPipeline(BaseEstimator):
    """
    Orquestador principal que coordina el entrenamiento de todas las familias
    de modelos de clasificación.
    
    Ejecuta en secuencia:
    1. Transformación del target a categorías
    2. Familia Árboles (RandomForest, ExtraTrees, GradientBoosting)
    3. Familia Lineales (LogisticRegression, SGDClassifier)
    4. Familia Boosting (XGBoost, LightGBM)
    
    NOTA: KNN ha sido eliminado del pipeline debido a overfitting severo (~27%)
    causado por lazy learning y curse of dimensionality con 66 features.
    
    Genera comparación automática y selecciona el mejor modelo por F1-score weighted.
    
    Ejemplo de uso:
        >>> pipeline = ClassificationTrainingPipeline(
        ...     n_bins=5,
        ...     binning_strategy='quantile',
        ...     cv_folds=5,
        ...     families=['trees', 'linear', 'boosting']
        ... )
        >>> pipeline.fit(X_train, y_train, X_val, y_val)
        >>> best_model, best_info = pipeline.get_best_model()
        >>> predictions = pipeline.predict(X_test)
    
    Args:
        n_bins: Número de categorías para el target (default: 5)
        binning_strategy: Estrategia de binning ('quantile', 'uniform', 'custom')
        cv_folds: Número de folds para cross-validation (default: 5)
        random_state: Semilla para reproducibilidad (default: 42)
        n_jobs: Número de cores a usar (-1 usa todos) (default: -1)
        verbose: Mostrar información detallada (default: True)
        families: Lista de familias a entrenar (disponibles: 'trees', 'linear', 'boosting')
        custom_bins: Lista de umbrales para binning personalizado (default: None)
        custom_labels: Lista de etiquetas personalizadas (default: None)
    """
    
    def __init__(self,
                 n_bins: int = 5,
                 binning_strategy: str = 'quantile',
                 cv_folds: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 families: List[str] = ['trees', 'knn', 'linear', 'boosting'],
                 custom_bins: Optional[List[float]] = None,
                 custom_labels: Optional[List[str]] = None):
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.families_to_train = [f.lower() for f in families]
        self.custom_bins = custom_bins
        self.custom_labels = custom_labels
        
        # Componentes
        self.target_transformer = None
        self.tree_family = None
        self.knn_family = None
        self.linear_family = None
        self.boosting_family = None
        
        # Datos de test (para evaluación final)
        self.X_test = None
        self.y_test = None
        self.y_test_cat = None
        
        # Resultados
        self.all_results = {}
        self.comparison_df = None
        self.test_comparison_df = None
        self.best_model_info = None
        self.best_test_model_info = None
        self.total_time = 0
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None) -> 'ClassificationTrainingPipeline':
        """
        Entrena todas las familias de modelos de clasificación.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento (Rating continuo)
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            X_test: Features de test (opcional)
            y_test: Target de test (opcional)
            
        Returns:
            self: Instancia entrenada
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(" CLASSIFICATION PIPELINE - INICIO ".center(80, "="))
            print("=" * 80)
            print(f"\nDimensiones de datos:")
            print(f"  Train: {X_train.shape}")
            if X_val is not None:
                print(f"  Val:   {X_val.shape}")
            if X_test is not None:
                print(f"  Test:  {X_test.shape}")
            print(f"  CV Folds: {self.cv_folds}")
            print(f"  Random State: {self.random_state}")
            print(f"  Familias a entrenar: {[f.upper() for f in self.families_to_train]}")
        
        start_total = time.time()
        
        # ====================================================================
        # TRANSFORMACIÓN DEL TARGET
        # ====================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print(" TRANSFORMANDO TARGET A CATEGORÍAS ".center(80, "="))
            print("=" * 80)
        
        self.target_transformer = TargetTransformer(
            n_bins=self.n_bins,
            strategy=self.binning_strategy,
            custom_bins=self.custom_bins,
            labels=self.custom_labels,
            verbose=self.verbose
        )
        
        y_train_cat = self.target_transformer.fit_transform(y_train)
        y_val_cat = self.target_transformer.transform(y_val) if y_val is not None else None
        
        # Guardar datos de validación para threshold optimization
        if X_val is not None and y_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.y_val_cat = y_val_cat
        
        # Guardar datos de test para evaluación final
        if X_test is not None and y_test is not None:
            self.X_test = X_test
            self.y_test = y_test
            self.y_test_cat = self.target_transformer.transform(y_test)
        
        # Validar distribución de clases
        class_dist = self.target_transformer.get_class_distribution(y_train_cat)
        min_samples = class_dist['Count'].min()
        
        if min_samples < 2:
            raise ValueError(
                f"Clase con muy pocas muestras ({min_samples}). "
                f"Reducir n_bins o fusionar clases raras."
            )
        
        if self.verbose:
            print(f"\n✓ Distribución de clases validada (mínimo {min_samples} muestras por clase)")
        
        # ====================================================================
        # CALCULAR CLASS WEIGHTS PARA BALANCEO
        # ====================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print(" CALCULANDO CLASS WEIGHTS PARA BALANCEO ".center(80, "="))
            print("=" * 80)
        
        # Calcular class_weight: penalizar más errores en clase minoritaria
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_cat),
            y=y_train_cat
        )
        
        # Crear diccionario {clase: peso}
        self.class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        if self.verbose:
            print(f"\nClass weights calculados (balanced):")
            for class_idx, weight in self.class_weight_dict.items():
                label = self.target_transformer.label_mapping_[class_idx]
                count = (y_train_cat == class_idx).sum()
                pct = count / len(y_train_cat) * 100
                print(f"  {label} (clase {class_idx}): weight={weight:.2f} | {count} muestras ({pct:.1f}%)")
            
            # Mostrar pesos más agresivos que usaremos
            print(f"\n✓ Usaremos class_weight MÁS AGRESIVO:")
            print(f"  Low (clase 0): weight=2.5x (vs {self.class_weight_dict[0]:.2f}x balanced)")
            print(f"  High (clase 1): weight=1.0x")
            print(f"  → Los modelos penalizarán 2.5x más los errores en Low")
        
        # ====================================================================
        # ENTRENAR FAMILIAS
        # ====================================================================
        family_count = 0
        total_families = len(self.families_to_train)
        
        # FAMILIA 1: ÁRBOLES
        if 'trees' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA ÁRBOLES ".center(80, "="))
                print("=" * 80)
            
            try:
                self.tree_family = TreeClassifierFamily(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                self.tree_family.fit(X_train, y_train_cat, X_val, y_val_cat)
                self.all_results['Trees'] = self.tree_family.get_results()
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ ERROR en familia Árboles: {str(e)}")
                    print("  Continuando con otras familias...")
        
        # FAMILIA 2: KNN - COMENTADO (overfitting 27% por lazy learning)
        # if 'knn' in self.families_to_train:
        #     family_count += 1
        #     if self.verbose:
        #         print("\n" + "=" * 80)
        #         print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA KNN ".center(80, "="))
        #         print("=" * 80)
        #     
        #     try:
        #         self.knn_family = KNNClassifierFamily(
        #             cv_folds=self.cv_folds,
        #             random_state=self.random_state,
        #             n_jobs=self.n_jobs,
        #             verbose=self.verbose
        #         )
        #         self.knn_family.fit(X_train, y_train_cat, X_val, y_val_cat)
        #         self.all_results['KNN'] = self.knn_family.get_results()
        #     except Exception as e:
        #         if self.verbose:
        #             print(f"\n⚠ ERROR en familia KNN: {str(e)}")
        #             print("  Continuando con otras familias...")
        
        # FAMILIA 3: LINEALES
        if 'linear' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA LINEALES ".center(80, "="))
                print("=" * 80)
            
            try:
                self.linear_family = LinearClassifierFamily(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                self.linear_family.fit(X_train, y_train_cat, X_val, y_val_cat)
                self.all_results['Linear'] = self.linear_family.get_results()
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ ERROR en familia Lineales: {str(e)}")
                    print("  Continuando con otras familias...")
        
        # FAMILIA 4: BOOSTING
        if 'boosting' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA BOOSTING ".center(80, "="))
                print("=" * 80)
            
            try:
                self.boosting_family = BoostingClassifierFamily(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                self.boosting_family.fit(X_train, y_train_cat, X_val, y_val_cat)
                self.all_results['Boosting'] = self.boosting_family.get_results()
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ ERROR en familia Boosting: {str(e)}")
                    print("  Continuando con otras familias...")
        
        self.total_time = time.time() - start_total
        
        # ====================================================================
        # GENERAR COMPARACIÓN EN VALIDACIÓN
        # ====================================================================
        self._generate_comparison()
        
        # ====================================================================
        # EVALUAR EN TEST (si está disponible)
        # ====================================================================
        if self.X_test is not None and self.y_test_cat is not None:
            self._evaluate_on_test(X_train, y_train_cat)
        
        if self.verbose:
            self._print_summary()
        
        return self
    
    def _optimize_threshold(self, y_true, y_proba, metric='f1_macro'):
        """
        Encuentra el threshold óptimo para clasificación binaria.
        
        Args:
            y_true: Etiquetas reales
            y_proba: Probabilidades predichas (columna 1 = clase positiva)
            metric: Métrica a optimizar ('f1_macro', 'recall_low', 'f1_weighted')
            
        Returns:
            float: Threshold óptimo
        """
        thresholds = np.arange(0.60, 0.95, 0.01)  # Rango alto para favorecer precision (0.60-0.95)
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
            
            if metric == 'f1_macro':
                score = f1_score(y_true, y_pred, average='macro', zero_division=0)
            elif metric == 'recall_low':
                # Recall de la clase Low (índice 0)
                recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                score = recall_per_class[0] if len(recall_per_class) > 0 else 0
            elif metric == 'f1_weighted':
                score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                score = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_on_test(self, X_train, y_train_cat):
        """Evalúa todos los modelos en el conjunto de test con threshold optimization"""
        if self.verbose:
            print("\n" + "=" * 80)
            print(" EVALUACIÓN EN TEST CON THRESHOLD OPTIMIZATION ".center(80, "="))
            print("=" * 80)
        
        test_results = []
        
        # Evaluar cada modelo en test
        for _, row in self.comparison_df.iterrows():
            familia = row['Familia']
            modelo = row['Modelo']
            
            # Obtener probabilidades en validación para encontrar threshold óptimo
            if familia == 'Árboles':
                y_proba_val = self.tree_family.predict_proba(self.X_val, model_name=modelo)
                y_proba_test = self.tree_family.predict_proba(self.X_test, model_name=modelo)
                y_proba_train = self.tree_family.predict_proba(X_train, model_name=modelo)
            elif familia == 'Lineales':
                y_proba_val = self.linear_family.predict_proba(self.X_val, model_name=modelo)
                y_proba_test = self.linear_family.predict_proba(self.X_test, model_name=modelo)
                y_proba_train = self.linear_family.predict_proba(X_train, model_name=modelo)
            elif familia == 'Boosting':
                y_proba_val = self.boosting_family.predict_proba(self.X_val, model_name=modelo)
                y_proba_test = self.boosting_family.predict_proba(self.X_test, model_name=modelo)
                y_proba_train = self.boosting_family.predict_proba(X_train, model_name=modelo)
            
            # Optimizar threshold en validación
            optimal_threshold = self._optimize_threshold(self.y_val_cat, y_proba_val, metric='f1_macro')
            
            if self.verbose:
                print(f"\n{modelo} ({familia}): threshold óptimo = {optimal_threshold:.3f}")
            
            # Aplicar threshold óptimo para hacer predicciones
            y_pred_test = (y_proba_test[:, 1] >= optimal_threshold).astype(int)
            y_pred_train = (y_proba_train[:, 1] >= optimal_threshold).astype(int)
            
            # Calcular métricas
            acc_train = accuracy_score(y_train_cat, y_pred_train)
            acc_test = accuracy_score(self.y_test_cat, y_pred_test)
            
            precision_test = precision_score(self.y_test_cat, y_pred_test, average='weighted', zero_division=0)
            recall_test = recall_score(self.y_test_cat, y_pred_test, average='weighted', zero_division=0)
            f1_weighted_test = f1_score(self.y_test_cat, y_pred_test, average='weighted', zero_division=0)
            f1_macro_test = f1_score(self.y_test_cat, y_pred_test, average='macro', zero_division=0)
            
            # Métricas por clase
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                self.y_test_cat, y_pred_test, zero_division=0
            )
            
            test_results.append({
                'Familia': familia,
                'Modelo': modelo,
                'Optimal_Threshold': optimal_threshold,
                'Accuracy_train': acc_train,
                'Accuracy_test': acc_test,
                'Precision_test': precision_test,
                'Recall_test': recall_test,
                'F1_weighted_test': f1_weighted_test,
                'F1_macro_test': f1_macro_test,
                'Overfitting_train_test': abs(acc_train - acc_test),
                'Tiempo_seg': row['Tiempo_seg'],
                'predictions': y_pred_test,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class
            })
        
        # Crear DataFrame y ordenar por F1-macro en test
        self.test_comparison_df = pd.DataFrame(test_results)
        self.test_comparison_df = self.test_comparison_df.sort_values('F1_macro_test', ascending=False).reset_index(drop=True)
        
        # Identificar mejor modelo en test
        if len(self.test_comparison_df) > 0:
            best_idx = self.test_comparison_df['F1_macro_test'].idxmax()
            self.best_test_model_info = self.test_comparison_df.loc[best_idx].to_dict()
    
    def _generate_comparison(self):
        """Genera tabla comparativa de todos los modelos"""
        comparison_data = []
        
        # Árboles
        if 'Trees' in self.all_results:
            for model_name in ['Random Forest', 'Extra Trees', 'Gradient Boosting']:
                if model_name in self.all_results['Trees']:
                    r = self.all_results['Trees'][model_name]
                    comparison_data.append({
                        'Familia': 'Árboles',
                        'Modelo': model_name,
                        'Accuracy_val': r['accuracy_val'],
                        'Precision_val': r['precision_val'],
                        'Recall_val': r['recall_val'],
                        'F1_weighted_val': r['f1_val'],
                        'F1_macro_val': r['f1_macro_val'],
                        'Accuracy_train': r['accuracy_train'],
                        'Overfitting': abs(r['accuracy_train'] - r['accuracy_val']),
                        'Tiempo_seg': r['train_time']
                    })
        
        # KNN - COMENTADO (no se entrena más)
        # if 'KNN' in self.all_results and 'KNN' in self.all_results['KNN']:
        #     r = self.all_results['KNN']['KNN']
        #     comparison_data.append({
        #         'Familia': 'KNN',
        #         'Modelo': 'KNN',
        #         'Accuracy_val': r['accuracy_val'],
        #         'Precision_val': r['precision_val'],
        #         'Recall_val': r['recall_val'],
        #         'F1_weighted_val': r['f1_val'],
        #         'F1_macro_val': r['f1_macro_val'],
        #         'Accuracy_train': r['accuracy_train'],
        #         'Overfitting': abs(r['accuracy_train'] - r['accuracy_val']),
        #         'Tiempo_seg': r['train_time']
        #     })
        
        # Lineales
        if 'Linear' in self.all_results:
            for model_name in ['Logistic Regression', 'SGD Classifier']:
                if model_name in self.all_results['Linear']:
                    r = self.all_results['Linear'][model_name]
                    comparison_data.append({
                        'Familia': 'Lineales',
                        'Modelo': model_name,
                        'Accuracy_val': r['accuracy_val'],
                        'Precision_val': r['precision_val'],
                        'Recall_val': r['recall_val'],
                        'F1_weighted_val': r['f1_val'],
                        'F1_macro_val': r['f1_macro_val'],
                        'Accuracy_train': r['accuracy_train'],
                        'Overfitting': abs(r['accuracy_train'] - r['accuracy_val']),
                        'Tiempo_seg': r['train_time']
                    })
        
        # Boosting
        if 'Boosting' in self.all_results:
            for model_name in ['XGBoost', 'CatBoost', 'LightGBM']:
                if model_name in self.all_results['Boosting']:
                    r = self.all_results['Boosting'][model_name]
                    comparison_data.append({
                        'Familia': 'Boosting',
                        'Modelo': model_name,
                        'Accuracy_val': r['accuracy_val'],
                        'Precision_val': r['precision_val'],
                        'Recall_val': r['recall_val'],
                        'F1_weighted_val': r['f1_val'],
                        'F1_macro_val': r['f1_macro_val'],
                        'Accuracy_train': r['accuracy_train'],
                        'Overfitting': abs(r['accuracy_train'] - r['accuracy_val']),
                        'Tiempo_seg': r['train_time']
                    })
        
        # Crear DataFrame y ordenar por F1_weighted_val
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('F1_weighted_val', ascending=False).reset_index(drop=True)
        
        # Identificar mejor modelo
        if len(self.comparison_df) > 0:
            best_idx = self.comparison_df['F1_weighted_val'].idxmax()
            self.best_model_info = self.comparison_df.loc[best_idx].to_dict()
    
    def _print_summary(self):
        """Imprime resumen comparativo con validación y test"""
        print("\n" + "=" * 80)
        print(" RESUMEN - VALIDACIÓN ".center(80, "="))
        print("=" * 80)
        
        print("\n📊 Ranking por F1-weighted en validación:")
        print(self.comparison_df.to_string(index=False))
        
        # Si hay resultados de test, mostrarlos
        if self.test_comparison_df is not None and len(self.test_comparison_df) > 0:
            print("\n" + "=" * 80)
            print(" RESUMEN - TEST (con Threshold Optimization) ".center(80, "="))
            print("=" * 80)
            
            # Mostrar solo columnas relevantes para test (incluyendo threshold)
            test_display_df = self.test_comparison_df[[
                'Familia', 'Modelo', 'Optimal_Threshold', 'Accuracy_train', 'Accuracy_test', 
                'F1_weighted_test', 'F1_macro_test', 'Overfitting_train_test', 'Tiempo_seg'
            ]].copy()
            
            print("\n📊 Ranking por F1-macro en test:")
            print(test_display_df.to_string(index=False))
            
            # MEJOR MODELO EN TEST
            if self.best_test_model_info:
                print("\n" + "=" * 80)
                print(" 🏆 MEJOR MODELO EN TEST ".center(80, "="))
                print("=" * 80)
                
                print(f"\n🎯 {self.best_test_model_info['Modelo']} ({self.best_test_model_info['Familia']})")
                print(f"   └─ Threshold óptimo:     {self.best_test_model_info['Optimal_Threshold']:.3f}")
                
                print(f"\n📊 Métricas Generales:")
                print(f"   ├─ Accuracy (train):     {self.best_test_model_info['Accuracy_train']:.4f}")
                print(f"   ├─ Accuracy (test):      {self.best_test_model_info['Accuracy_test']:.4f}")
                print(f"   ├─ Precision (test):     {self.best_test_model_info['Precision_test']:.4f}")
                print(f"   ├─ Recall (test):        {self.best_test_model_info['Recall_test']:.4f}")
                print(f"   ├─ F1-weighted (test):   {self.best_test_model_info['F1_weighted_test']:.4f}")
                print(f"   └─ F1-macro (test):      {self.best_test_model_info['F1_macro_test']:.4f}")
                
                print(f"\n📉 Overfitting:")
                ovf_pct = self.best_test_model_info['Overfitting_train_test'] * 100
                print(f"   └─ Train-Test Gap:       {self.best_test_model_info['Overfitting_train_test']:.4f} ({ovf_pct:.2f}%)")
                
                # Métricas por clase
                print(f"\n📊 Métricas por Clase:")
                for i, label in enumerate(self.target_transformer.labels):
                    precision_class = self.best_test_model_info['precision_per_class'][i]
                    recall_class = self.best_test_model_info['recall_per_class'][i]
                    f1_class = self.best_test_model_info['f1_per_class'][i]
                    
                    print(f"\n   {label}:")
                    print(f"      ├─ Precision:  {precision_class:.4f}")
                    print(f"      ├─ Recall:     {recall_class:.4f}")
                    print(f"      └─ F1-score:   {f1_class:.4f}")
                
                # Matriz de confusión
                y_pred_best = self.best_test_model_info['predictions']
                cm = confusion_matrix(self.y_test_cat, y_pred_best)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                print(f"\n📊 Matriz de Confusión (conteos):")
                print(f"   {'':>10}", end='')
                for label in self.target_transformer.labels:
                    print(f"  Pred_{label:>6}", end='')
                print()
                for i, label in enumerate(self.target_transformer.labels):
                    print(f"   Real_{label:>5}", end='')
                    for j in range(len(self.target_transformer.labels)):
                        print(f"  {cm[i, j]:>9}", end='')
                    print()
                
                print(f"\n📊 Matriz de Confusión (normalizada):")
                print(f"   {'':>10}", end='')
                for label in self.target_transformer.labels:
                    print(f"  Pred_{label:>6}", end='')
                print()
                for i, label in enumerate(self.target_transformer.labels):
                    print(f"   Real_{label:>5}", end='')
                    for j in range(len(self.target_transformer.labels)):
                        print(f"  {cm_normalized[i, j]:>9.4f}", end='')
                    print()
        
        print(f"\n{'=' * 80}")
        print(f"⏱️  Tiempo total: {self.total_time:.2f}s ({self.total_time/60:.2f} min)")
        print(f"{'=' * 80}")
    
    def predict(self, X, family='best', model_name=None) -> np.ndarray:
        """
        Realiza predicciones con el modelo especificado.
        
        Args:
            X: Features para predecir
            family: 'best', 'trees', 'linear', o 'boosting'
            model_name: Nombre específico del modelo (ej: 'Random Forest')
            
        Returns:
            np.ndarray: Predicciones de clase
        """
        if family == 'best':
            family = self.best_model_info['Familia'].lower()
            model_name = self.best_model_info['Modelo']
        
        if 'árbol' in family.lower() or 'tree' in family.lower():
            return self.tree_family.predict(X, model_name=model_name)
        elif 'linear' in family.lower():
            return self.linear_family.predict(X, model_name=model_name)
        elif 'boost' in family.lower():
            return self.boosting_family.predict(X, model_name=model_name)
        else:
            raise ValueError(f"Familia desconocida: {family}")
    
    def predict_proba(self, X, family='best', model_name=None) -> np.ndarray:
        """Predice probabilidades con el modelo especificado"""
        if family == 'best':
            family = self.best_model_info['Familia'].lower()
            model_name = self.best_model_info['Modelo']
        
        if 'árbol' in family.lower() or 'tree' in family.lower():
            return self.tree_family.predict_proba(X, model_name=model_name)
        elif 'linear' in family.lower():
            return self.linear_family.predict_proba(X, model_name=model_name)
        elif 'boost' in family.lower():
            return self.boosting_family.predict_proba(X, model_name=model_name)
        else:
            raise ValueError(f"Familia desconocida: {family}")
    
    def get_best_model(self) -> Tuple[BaseEstimator, Dict]:
        """
        Retorna el mejor modelo entrenado.
        
        Returns:
            tuple: (modelo, información del modelo)
        """
        family = self.best_model_info['Familia']
        model_name = self.best_model_info['Modelo']
        
        if family == 'Árboles':
            return self.all_results['Trees'][model_name]['model'], self.best_model_info
        # elif family == 'KNN':  # COMENTADO - KNN ya no se entrena
        #     return self.all_results['KNN']['KNN']['model'], self.best_model_info
        elif family == 'Lineales':
            return self.all_results['Linear'][model_name]['model'], self.best_model_info
        elif family == 'Boosting':
            return self.all_results['Boosting'][model_name]['model'], self.best_model_info
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Retorna DataFrame con comparación de todos los modelos"""
        return self.comparison_df
    
    def get_all_results(self) -> Dict:
        """Retorna diccionario con todos los resultados detallados"""
        return self.all_results
    
    def save_results(self, output_dir: str = '.'):
        """
        Guarda todos los resultados en archivos.
        
        Args:
            output_dir: Directorio donde guardar los archivos
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Guardar comparación de validación
        comparison_path = output_dir / 'model_comparison_validation.csv'
        self.comparison_df.to_csv(comparison_path, index=False)
        
        # Guardar comparación de test (si existe)
        if self.test_comparison_df is not None:
            test_comparison_path = output_dir / 'model_comparison_test.csv'
            # Guardar sin las columnas de arrays (predictions, etc)
            test_df_to_save = self.test_comparison_df.drop(columns=['predictions', 'precision_per_class', 'recall_per_class', 'f1_per_class'])
            test_df_to_save.to_csv(test_comparison_path, index=False)
        
        # Guardar matriz de confusión del mejor modelo (si existe)
        if self.best_test_model_info is not None:
            y_pred_best = self.best_test_model_info['predictions']
            cm = confusion_matrix(self.y_test_cat, y_pred_best)
            cm_df = pd.DataFrame(cm,
                                index=[f'Real_{label}' for label in self.target_transformer.labels],
                                columns=[f'Pred_{label}' for label in self.target_transformer.labels])
            cm_path = output_dir / 'confusion_matrix_best_model.csv'
            cm_df.to_csv(cm_path)
        
        # Guardar resumen en texto
        summary_path = output_dir / 'training_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" RESUMEN DE ENTRENAMIENTO - CLASIFICACIÓN ".center(80) + "\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPARACIÓN DE MODELOS (VALIDACIÓN):\n")
            f.write(self.comparison_df.to_string(index=False) + "\n\n")
            
            if self.test_comparison_df is not None:
                f.write("\nCOMPARACIÓN DE MODELOS (TEST):\n")
                test_df_display = self.test_comparison_df[[
                    'Familia', 'Modelo', 'Accuracy_train', 'Accuracy_test',
                    'F1_weighted_test', 'F1_macro_test', 'Overfitting_train_test', 'Tiempo_seg'
                ]]
                f.write(test_df_display.to_string(index=False) + "\n\n")
            
            if self.best_test_model_info:
                f.write("\nMEJOR MODELO (TEST):\n")
                f.write(f"  Familia: {self.best_test_model_info['Familia']}\n")
                f.write(f"  Modelo: {self.best_test_model_info['Modelo']}\n")
                f.write(f"  Accuracy (train): {self.best_test_model_info['Accuracy_train']:.4f}\n")
                f.write(f"  Accuracy (test): {self.best_test_model_info['Accuracy_test']:.4f}\n")
                f.write(f"  F1-weighted (test): {self.best_test_model_info['F1_weighted_test']:.4f}\n")
                f.write(f"  F1-macro (test): {self.best_test_model_info['F1_macro_test']:.4f}\n")
                f.write(f"  Overfitting: {self.best_test_model_info['Overfitting_train_test']:.4f}\n")
            elif self.best_model_info:
                f.write("\nMEJOR MODELO (VALIDACIÓN):\n")
                for key, value in self.best_model_info.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTiempo total: {self.total_time:.2f}s ({self.total_time/60:.2f} min)\n")
            
            f.write(f"\nBINS DE RATING:\n")
            for i, edge in enumerate(self.target_transformer.get_bin_edges()[:-1]):
                next_edge = self.target_transformer.get_bin_edges()[i+1]
                label = self.target_transformer.labels[i]
                f.write(f"  {label}: [{edge:.3f}, {next_edge:.3f})\n")
        
        if self.verbose:
            print(f"\n✓ Resultados guardados en: {output_dir}")
            print(f"  ├─ {comparison_path.name}")
            if self.test_comparison_df is not None:
                print(f"  ├─ model_comparison_test.csv")
                if self.best_test_model_info is not None:
                    print(f"  ├─ confusion_matrix_best_model.csv")
            print(f"  └─ {summary_path.name}")
    
    def plot_confusion_matrix(self, X, y, family='best', normalize=True, figsize=(10, 8)):
        """
        Genera y visualiza matriz de confusión.
        
        Args:
            X: Features
            y: Target (categórico)
            family: Familia del modelo a usar
            normalize: Normalizar matriz (default: True)
            figsize: Tamaño de la figura
        """
        # Obtener predicciones
        y_pred = self.predict(X, family=family)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualizar
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=self.target_transformer.labels,
                   yticklabels=self.target_transformer.labels)
        plt.ylabel('Clase Real')
        plt.xlabel('Clase Predicha')
        plt.title(f'Matriz de Confusión - {self.best_model_info["Modelo"]}')
        plt.tight_layout()
        plt.show()
    
    def get_classification_report(self, X, y, family='best') -> str:
        """
        Genera reporte de clasificación completo.
        
        Args:
            X: Features
            y: Target (categórico)
            family: Familia del modelo a usar
            
        Returns:
            str: Reporte de clasificación
        """
        y_pred = self.predict(X, family=family)
        return classification_report(y, y_pred, target_names=self.target_transformer.labels, zero_division=0)