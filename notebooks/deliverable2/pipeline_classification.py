"""
Classification Pipeline for Google Play Store Rating Prediction

Este módulo implementa un pipeline completo de clasificación multiclase para predecir
categorías de rating de aplicaciones en Google Play Store.

Familias de modelos incluidas:
- Árboles: RandomForest, ExtraTrees, GradientBoosting
- KNN: K-Nearest Neighbors
- Lineales: LogisticRegression, SGDClassifier
- Boosting: XGBoost, LightGBM

Autor: Sistema de ML
Fecha: 2025
Versión: 1.0.0

Dependencias:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.1
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

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
            pd.Series: Serie con categorías
        """
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() antes de transform()")
        
        # Usar pd.cut para binning
        y_binned = pd.cut(
            y,
            bins=self.bin_edges_,
            labels=self.labels,
            include_lowest=True,
            duplicates='drop'
        )
        
        # Calcular distribución si es verbose
        if self.verbose:
            dist = y_binned.value_counts().sort_index()
            print(f"\nDistribución de clases:")
            for label, count in dist.items():
                pct = count / len(y) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")
        
        return y_binned
    
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
            y: Serie con categorías (después de transform)
            
        Returns:
            pd.DataFrame: Distribución con conteos y porcentajes
        """
        dist = y.value_counts().sort_index()
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
        
        # ====================================================================
        # RANDOM FOREST
        # ====================================================================
        if self.verbose:
            print("\n[1/3] Entrenando Random Forest...")
            print("-" * 80)
        
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        if self.verbose:
            total_rf = (len(param_grid_rf['n_estimators']) * 
                       len(param_grid_rf['max_depth']) * 
                       len(param_grid_rf['min_samples_split']) *
                       len(param_grid_rf['min_samples_leaf']) *
                       len(param_grid_rf['max_features']))
            print(f"  Combinaciones: {total_rf}")
        
        start_time = time.time()
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
            ),
            param_grid_rf,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        rf_grid.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        self.models['Random Forest'] = rf_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {rf_time:.2f}s")
            print(f"  Mejores parámetros: {rf_grid.best_params_}")
            print(f"  Mejor CV F1-score: {rf_grid.best_score_:.4f}")
        
        # Evaluar en validación
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
        
        et_model = ExtraTreesClassifier(
            n_estimators=best_rf_params['n_estimators'],
            max_depth=best_rf_params['max_depth'],
            min_samples_split=best_rf_params['min_samples_split'],
            min_samples_leaf=best_rf_params['min_samples_leaf'],
            max_features=best_rf_params['max_features'],
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=self.n_jobs
        )
        
        et_model.fit(X_train, y_train)
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
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3, 4],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }
        
        if self.verbose:
            total_gb = (len(param_grid_gb['learning_rate']) * 
                       len(param_grid_gb['n_estimators']) * 
                       len(param_grid_gb['max_depth']) *
                       len(param_grid_gb['subsample']) *
                       len(param_grid_gb['min_samples_split']) *
                       len(param_grid_gb['min_samples_leaf']))
            print(f"  Combinaciones: {total_gb}")
        
        start_time = time.time()
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_state),
            param_grid_gb,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        gb_grid.fit(X_train, y_train)
        gb_time = time.time() - start_time
        
        self.models['Gradient Boosting'] = gb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {gb_time:.2f}s")
            print(f"  Mejores parámetros: {gb_grid.best_params_}")
            print(f"  Mejor CV F1-score: {gb_grid.best_score_:.4f}")
        
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
# FAMILIA KNN: K-Nearest Neighbors
# ============================================================================

class KNNClassifierFamily(BaseEstimator):
    """
    Familia K-Nearest Neighbors con escalado StandardScaler.
    
    KNN es sensible a la escala de las features, por lo que requiere
    normalización. Se optimizan hiperparámetros con GridSearchCV.
    
    Ejemplo de uso:
        >>> knn_family = KNNClassifierFamily(cv_folds=5, random_state=42)
        >>> knn_family.fit(X_train, y_train, X_val, y_val)
        >>> predictions = knn_family.predict(X_test)
    
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
        self.model = None
        self.results = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'KNNClassifierFamily':
        """Entrena KNN con escalado y optimización de hiperparámetros"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA KNN: K-NEAREST NEIGHBORS".center(80))
            print("=" * 80)
        
        # Escalado
        if self.verbose:
            print("\n[1/2] Aplicando StandardScaler...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        if self.verbose:
            print(f"  ✓ Datos escalados: μ={X_train_scaled.mean():.6f}, σ={X_train_scaled.std():.6f}")
        
        # GridSearchCV
        if self.verbose:
            print("\n[2/2] Entrenando KNN con GridSearchCV...")
            print("-" * 80)
        
        param_grid_knn = {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree']
        }
        
        if self.verbose:
            total_knn = (len(param_grid_knn['n_neighbors']) * 
                        len(param_grid_knn['weights']) * 
                        len(param_grid_knn['metric']) *
                        len(param_grid_knn['algorithm']))
            print(f"  Combinaciones: {total_knn}")
        
        start_time = time.time()
        
        knn_grid = GridSearchCV(
            KNeighborsClassifier(n_jobs=self.n_jobs),
            param_grid_knn,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        knn_grid.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        self.model = knn_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {train_time:.2f}s")
            print(f"  Mejores parámetros: {knn_grid.best_params_}")
            print(f"  Mejor CV F1-score: {knn_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
            # Medir tiempo de predicción
            pred_start = time.time()
            y_pred_val = knn_grid.predict(X_val_scaled)
            pred_time = time.time() - pred_start
            
            y_pred_train = knn_grid.predict(X_train_scaled)
            y_proba_val = knn_grid.predict_proba(X_val_scaled)
            
            self.results['KNN'] = {
                'model': knn_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': knn_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': train_time,
                'pred_time': pred_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
            
            if self.verbose:
                print(f"\n  Métricas en validación:")
                print(f"    Accuracy:  {self.results['KNN']['accuracy_val']:.4f}")
                print(f"    F1-score:  {self.results['KNN']['f1_val']:.4f}")
                print(f"    Tiempo de predicción: {pred_time:.4f}s")
                
                # Advertencia si predicción es lenta
                if pred_time > 1.0:
                    print(f"\n  ⚠ ADVERTENCIA: Tiempo de predicción alto ({pred_time:.2f}s)")
                    print(f"    Considerar usar algorithm='ball_tree' o reducir n_neighbors")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predice clases (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predice probabilidades (escala datos automáticamente)"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_results(self) -> Dict:
        """Retorna diccionario con todos los resultados"""
        return self.results



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
        
        # Escalado
        if self.verbose:
            print("\n[0/2] Aplicando StandardScaler...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
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
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        lr_grid.fit(X_train_scaled, y_train)
        lr_time = time.time() - start_time
        
        self.models['Logistic Regression'] = lr_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {lr_time:.2f}s")
            print(f"  Mejores parámetros: {lr_grid.best_params_}")
            print(f"  Mejor CV F1-score: {lr_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
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
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        sgd_grid.fit(X_train_scaled, y_train)
        sgd_time = time.time() - start_time
        
        self.models['SGD Classifier'] = sgd_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {sgd_time:.2f}s")
            print(f"  Mejores parámetros: {sgd_grid.best_params_}")
            print(f"  Mejor CV F1-score: {sgd_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
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
# FAMILIA BOOSTING: XGBoost, LightGBM
# ============================================================================

class BoostingClassifierFamily(BaseEstimator):
    """
    Familia de modelos de boosting avanzados para clasificación.
    
    Incluye XGBoost y LightGBM con MinMaxScaler y early stopping.
    Ambos modelos manejan desbalance de clases automáticamente.
    
    Ejemplo de uso:
        >>> boosting_family = BoostingClassifierFamily(cv_folds=5, random_state=42)
        >>> boosting_family.fit(X_train, y_train, X_val, y_val)
        >>> predictions = boosting_family.predict(X_test, model_name='LightGBM')
    
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
            print("FAMILIA BOOSTING: XGBOOST, LIGHTGBM".center(80))
            print("=" * 80)
        
        # Escalado
        if self.verbose:
            print("\n[0/2] Aplicando MinMaxScaler...")
        
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        if self.verbose:
            print(f"  ✓ Datos escalados: rango=[{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
        
        # ====================================================================
        # XGBOOST
        # ====================================================================
        if self.verbose:
            print("\n[1/2] Entrenando XGBoost...")
            print("-" * 80)
        
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [0.5, 1.0],
            'reg_lambda': [0.5, 1.0]
        }
        
        if self.verbose:
            total_xgb = (len(param_grid_xgb['n_estimators']) * 
                        len(param_grid_xgb['learning_rate']) * 
                        len(param_grid_xgb['max_depth']) *
                        len(param_grid_xgb['subsample']) *
                        len(param_grid_xgb['colsample_bytree']) *
                        len(param_grid_xgb['reg_alpha']) *
                        len(param_grid_xgb['reg_lambda']))
            print(f"  Combinaciones: {total_xgb}")
        
        start_time = time.time()
        
        xgb_grid = GridSearchCV(
            XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='mlogloss'
            ),
            param_grid_xgb,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        xgb_grid.fit(X_train_scaled, y_train)
        xgb_time = time.time() - start_time
        
        self.models['XGBoost'] = xgb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {xgb_time:.2f}s")
            print(f"  Mejores parámetros: {xgb_grid.best_params_}")
            print(f"  Mejor CV F1-score: {xgb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
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
        # LIGHTGBM
        # ====================================================================
        if self.verbose:
            print("\n[2/2] Entrenando LightGBM...")
            print("-" * 80)
        
        param_grid_lgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'reg_alpha': [0.5, 1.0],
            'reg_lambda': [0.5, 1.0],
            'min_child_samples': [10, 20]
        }
        
        if self.verbose:
            total_lgb = (len(param_grid_lgb['n_estimators']) * 
                        len(param_grid_lgb['learning_rate']) * 
                        len(param_grid_lgb['max_depth']) *
                        len(param_grid_lgb['subsample']) *
                        len(param_grid_lgb['colsample_bytree']) *
                        len(param_grid_lgb['reg_alpha']) *
                        len(param_grid_lgb['reg_lambda']) *
                        len(param_grid_lgb['min_child_samples']))
            print(f"  Combinaciones: {total_lgb}")
        
        start_time = time.time()
        
        lgb_grid = GridSearchCV(
            LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight='balanced',
                verbose=-1
            ),
            param_grid_lgb,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        lgb_grid.fit(X_train_scaled, y_train)
        lgb_time = time.time() - start_time
        
        self.models['LightGBM'] = lgb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {lgb_time:.2f}s")
            print(f"  Mejores parámetros: {lgb_grid.best_params_}")
            print(f"  Mejor CV F1-score: {lgb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
            y_pred_val = lgb_grid.predict(X_val_scaled)
            y_pred_train = lgb_grid.predict(X_train_scaled)
            y_proba_val = lgb_grid.predict_proba(X_val_scaled)
            
            self.results['LightGBM'] = {
                'model': lgb_grid.best_estimator_,
                'scaler': self.scaler,
                'best_params': lgb_grid.best_params_,
                'accuracy_val': accuracy_score(y_val, y_pred_val),
                'precision_val': precision_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'recall_val': recall_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_val': f1_score(y_val, y_pred_val, average='weighted', zero_division=0),
                'f1_macro_val': f1_score(y_val, y_pred_val, average='macro', zero_division=0),
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'train_time': lgb_time,
                'predictions_val': y_pred_val,
                'probabilities_val': y_proba_val,
                'confusion_matrix': confusion_matrix(y_val, y_pred_val),
                'classification_report': classification_report(y_val, y_pred_val, zero_division=0)
            }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA BOOSTING".center(80))
            print(f"{'=' * 80}")
            for model_name in ['XGBoost', 'LightGBM']:
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
    3. Familia KNN (K-Nearest Neighbors)
    4. Familia Lineales (LogisticRegression, SGDClassifier)
    5. Familia Boosting (XGBoost, LightGBM)
    
    Genera comparación automática y selecciona el mejor modelo por F1-score weighted.
    
    Ejemplo de uso:
        >>> pipeline = ClassificationTrainingPipeline(
        ...     n_bins=5,
        ...     binning_strategy='quantile',
        ...     cv_folds=5,
        ...     families=['trees', 'knn', 'linear', 'boosting']
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
        families: Lista de familias a entrenar (default: todas)
    """
    
    def __init__(self,
                 n_bins: int = 5,
                 binning_strategy: str = 'quantile',
                 cv_folds: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 families: List[str] = ['trees', 'knn', 'linear', 'boosting']):
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.families_to_train = [f.lower() for f in families]
        
        # Componentes
        self.target_transformer = None
        self.tree_family = None
        self.knn_family = None
        self.linear_family = None
        self.boosting_family = None
        
        # Resultados
        self.all_results = {}
        self.comparison_df = None
        self.best_model_info = None
        self.total_time = 0
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> 'ClassificationTrainingPipeline':
        """
        Entrena todas las familias de modelos de clasificación.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento (Rating continuo)
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            
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
            verbose=self.verbose
        )
        
        y_train_cat = self.target_transformer.fit_transform(y_train)
        y_val_cat = self.target_transformer.transform(y_val) if y_val is not None else None
        
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
        
        # FAMILIA 2: KNN
        if 'knn' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA KNN ".center(80, "="))
                print("=" * 80)
            
            try:
                self.knn_family = KNNClassifierFamily(
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                self.knn_family.fit(X_train, y_train_cat, X_val, y_val_cat)
                self.all_results['KNN'] = self.knn_family.get_results()
            except Exception as e:
                if self.verbose:
                    print(f"\n⚠ ERROR en familia KNN: {str(e)}")
                    print("  Continuando con otras familias...")
        
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
        # GENERAR COMPARACIÓN
        # ====================================================================
        self._generate_comparison()
        
        if self.verbose:
            self._print_summary()
        
        return self
    
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
        
        # KNN
        if 'KNN' in self.all_results and 'KNN' in self.all_results['KNN']:
            r = self.all_results['KNN']['KNN']
            comparison_data.append({
                'Familia': 'KNN',
                'Modelo': 'KNN',
                'Accuracy_val': r['accuracy_val'],
                'Precision_val': r['precision_val'],
                'Recall_val': r['recall_val'],
                'F1_weighted_val': r['f1_val'],
                'F1_macro_val': r['f1_macro_val'],
                'Accuracy_train': r['accuracy_train'],
                'Overfitting': abs(r['accuracy_train'] - r['accuracy_val']),
                'Tiempo_seg': r['train_time']
            })
        
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
            for model_name in ['XGBoost', 'LightGBM']:
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
        
        # Crear DataFrame y ordenar por F1_weighted
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('F1_weighted_val', ascending=False).reset_index(drop=True)
        
        # Identificar mejor modelo
        if len(self.comparison_df) > 0:
            best_idx = self.comparison_df['F1_weighted_val'].idxmax()
            self.best_model_info = self.comparison_df.loc[best_idx].to_dict()
    
    def _print_summary(self):
        """Imprime resumen comparativo"""
        print("\n" + "=" * 80)
        print(" RESUMEN COMPARATIVO - TODOS LOS MODELOS ".center(80, "="))
        print("=" * 80)
        
        print("\nRanking por F1-score weighted en validación:")
        print(self.comparison_df.to_string(index=False))
        
        print(f"\n{'=' * 80}")
        print(f"Tiempo total de entrenamiento: {self.total_time:.2f}s ({self.total_time/60:.2f} min)")
        print(f"{'=' * 80}")
        
        if self.best_model_info:
            print(f"\n🏆 MEJOR MODELO:")
            print(f"   Familia:          {self.best_model_info['Familia']}")
            print(f"   Modelo:           {self.best_model_info['Modelo']}")
            print(f"   Accuracy (val):   {self.best_model_info['Accuracy_val']:.4f}")
            print(f"   F1-score (val):   {self.best_model_info['F1_weighted_val']:.4f}")
            print(f"   Precision (val):  {self.best_model_info['Precision_val']:.4f}")
            print(f"   Recall (val):     {self.best_model_info['Recall_val']:.4f}")
            print(f"   Overfitting:      {self.best_model_info['Overfitting']:.4f}")
            print(f"   Tiempo:           {self.best_model_info['Tiempo_seg']:.2f}s")
    
    def predict(self, X, family='best', model_name=None) -> np.ndarray:
        """
        Realiza predicciones con el modelo especificado.
        
        Args:
            X: Features para predecir
            family: 'best', 'trees', 'knn', 'linear', o 'boosting'
            model_name: Nombre específico del modelo (ej: 'Random Forest')
            
        Returns:
            np.ndarray: Predicciones de clase
        """
        if family == 'best':
            family = self.best_model_info['Familia'].lower()
            model_name = self.best_model_info['Modelo']
        
        if 'árbol' in family.lower() or 'tree' in family.lower():
            return self.tree_family.predict(X, model_name=model_name)
        elif 'knn' in family.lower():
            return self.knn_family.predict(X)
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
        elif 'knn' in family.lower():
            return self.knn_family.predict_proba(X)
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
        elif family == 'KNN':
            return self.all_results['KNN']['KNN']['model'], self.best_model_info
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
        
        # Guardar comparación
        comparison_path = output_dir / 'classification_model_comparison.csv'
        self.comparison_df.to_csv(comparison_path, index=False)
        
        # Guardar resumen en texto
        summary_path = output_dir / 'classification_training_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" RESUMEN DE ENTRENAMIENTO - CLASIFICACIÓN ".center(80) + "\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPARACIÓN DE MODELOS:\n")
            f.write(self.comparison_df.to_string(index=False) + "\n\n")
            
            f.write("MEJOR MODELO:\n")
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
            print(f"  - {comparison_path.name}")
            print(f"  - {summary_path.name}")
    
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
