import pandas as pd
import numpy as np
import time
import warnings
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# FAMILIA SVM CON BÚSQUEDA OPTIMIZADA EN 2 FASES
# ============================================================================

class SVMFamily(BaseEstimator):
    """
    Familia SVM con preprocesamiento StandardScaler y búsqueda optimizada.
    
    Estrategia de búsqueda en 2 fases:
    - Fase 1: Búsqueda rápida para identificar mejor kernel (9 combinaciones)
    - Fase 2: Refinamiento con el mejor kernel encontrado
    
    Esto reduce el tiempo de 720 fits a ~60-90 fits sin perder calidad.
    """
    
    def __init__(self, cv_folds=3, random_state=42, n_jobs=-1, verbose=True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Componentes
        self.scaler = None
        self.best_kernel = None
        self.phase1_grid = None
        self.phase2_grid = None
        self.best_model = None
        self.results = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena SVM con búsqueda optimizada en 2 fases"""
        if self.verbose:
            print("=" * 80)
            print("FAMILIA SVM: BÚSQUEDA OPTIMIZADA EN 2 FASES".center(80))
            print("=" * 80)
        
        # ====================================================================
        # PREPROCESAMIENTO: StandardScaler
        # ====================================================================
        if self.verbose:
            print("\n[1/3] Aplicando StandardScaler...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        if self.verbose:
            print(f"  ✓ Datos escalados: μ={X_train_scaled.mean():.6f}, σ={X_train_scaled.std():.6f}")
        
        # ====================================================================
        # FASE 1: Búsqueda rápida de mejor kernel
        # ====================================================================
        if self.verbose:
            print("\n[2/3] FASE 1: Búsqueda rápida de mejor kernel")
            print("-" * 80)
        
        param_grid_phase1 = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [1, 10, 50],
            'gamma': ['scale'],
            'epsilon': [0.1]
        }
        
        total_phase1 = (len(param_grid_phase1['kernel']) * 
                       len(param_grid_phase1['C']) * 
                       len(param_grid_phase1['gamma']) * 
                       len(param_grid_phase1['epsilon']))
        
        if self.verbose:
            print(f"  Combinaciones: {total_phase1} (vs 144 original)")
            print(f"  Total fits: {total_phase1 * self.cv_folds}")
        
        start_phase1 = time.time()
        
        self.phase1_grid = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid_phase1,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        self.phase1_grid.fit(X_train_scaled, y_train)
        phase1_time = time.time() - start_phase1
        
        self.best_kernel = self.phase1_grid.best_params_['kernel']
        
        if self.verbose:
            print(f"\n  ✓ Fase 1 completada en {phase1_time:.2f}s")
            print(f"  Mejor kernel: {self.best_kernel}")
            print(f"  Mejor score: {-self.phase1_grid.best_score_:.4f}")
        
        # ====================================================================
        # FASE 2: Refinamiento con mejor kernel
        # ====================================================================
        if self.verbose:
            print(f"\n[3/3] FASE 2: Refinamiento con kernel={self.best_kernel}")
            print("-" * 80)
        
        if self.best_kernel == 'linear':
            param_grid_phase2 = {
                'kernel': [self.best_kernel],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2]
            }
        else:
            param_grid_phase2 = {
                'kernel': [self.best_kernel],
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto', 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        
        if self.best_kernel == 'linear':
            total_phase2 = (len(param_grid_phase2['C']) * 
                           len(param_grid_phase2['epsilon']))
        else:
            total_phase2 = (len(param_grid_phase2['C']) * 
                           len(param_grid_phase2['gamma']) * 
                           len(param_grid_phase2['epsilon']))
        
        if self.verbose:
            print(f"  Combinaciones: {total_phase2}")
            print(f"  Total fits: {total_phase2 * self.cv_folds}")
        
        start_phase2 = time.time()
        
        self.phase2_grid = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid_phase2,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            return_train_score=True
        )
        
        self.phase2_grid.fit(X_train_scaled, y_train)
        phase2_time = time.time() - start_phase2
        
        self.best_model = self.phase2_grid.best_estimator_
        total_time = phase1_time + phase2_time
        
        if self.verbose:
            print(f"\n  ✓ Fase 2 completada en {phase2_time:.2f}s")
            print(f"\n{'=' * 80}")
            print("RESUMEN SVM".center(80))
            print(f"{'=' * 80}")
            print(f"  Tiempo total: {total_time:.2f}s ({total_time/60:.2f} min)")
            print(f"  Mejores parámetros:")
            for param, value in self.phase2_grid.best_params_.items():
                print(f"    - {param}: {value}")
            print(f"  Mejor CV MAE: {-self.phase2_grid.best_score_:.4f}")
        
        # Evaluar en validación si está disponible
        if X_val is not None and y_val is not None:
            y_pred_val = self.best_model.predict(X_val_scaled)
            y_pred_train = self.best_model.predict(X_train_scaled)
            
            self.results = {
                'model': self.best_model,
                'scaler': self.scaler,
                'best_params': self.phase2_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': total_time,
                'best_kernel': self.best_kernel,
                'phase1_time': phase1_time,
                'phase2_time': phase2_time,
                'predictions_val': y_pred_val
            }
            
            if self.verbose:
                print(f"\n  Métricas en validación:")
                print(f"    MAE:  {self.results['mae_val']:.4f}")
                print(f"    RMSE: {self.results['rmse_val']:.4f}")
                print(f"    R²:   {self.results['r2_val']:.4f}")
                print(f"  Overfitting: {abs(self.results['mae_train'] - self.results['mae_val']):.4f}")
        
        return self
    
    def predict(self, X):
        """Predice usando el mejor modelo encontrado"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_results(self):
        """Retorna diccionario con todos los resultados"""
        return self.results


# ============================================================================
# FAMILIA ÁRBOLES (Decision Tree, Random Forest, Extra Trees)
# ============================================================================

class TreeFamily(BaseEstimator):
    """
    Familia de modelos basados en árboles.
    
    No requiere escalado de datos. Incluye:
    - Decision Tree con optimización de hiperparámetros
    - Random Forest con GridSearchCV
    - Extra Trees (usa parámetros de Random Forest)
    """
    
    def __init__(self, cv_folds=3, random_state=42, n_jobs=-1, verbose=True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.models = {}
        self.results = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena todos los modelos de árboles"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA ÁRBOLES: DECISION TREE, RANDOM FOREST, EXTRA TREES".center(80))
            print("=" * 80)
        
        # ====================================================================
        # DECISION TREE
        # ====================================================================
        if self.verbose:
            print("\n[1/3] Entrenando Decision Tree...")
            print("-" * 80)
        
        param_grid_dt = {
            'max_depth': [3, 5, 8, 10],  # Limitar profundidad
            'min_samples_split': [10, 20, 50],  # Más regularización
            'min_samples_leaf': [5, 10, 20]  # Más regularización
        }
        
        if self.verbose:
            total_dt = (len(param_grid_dt['max_depth']) * 
                       len(param_grid_dt['min_samples_split']) * 
                       len(param_grid_dt['min_samples_leaf']))
            print(f"  Combinaciones: {total_dt}")
        
        start_time = time.time()
        
        dt_grid = GridSearchCV(
            DecisionTreeRegressor(random_state=self.random_state),
            param_grid_dt,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        dt_grid.fit(X_train, y_train)
        dt_time = time.time() - start_time
        
        self.models['Decision Tree'] = dt_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {dt_time:.2f}s")
            print(f"  Mejores parámetros: {dt_grid.best_params_}")
            print(f"  Mejor CV MAE: {-dt_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            y_pred_val = dt_grid.predict(X_val)
            y_pred_train = dt_grid.predict(X_train)
            
            self.results['Decision Tree'] = {
                'model': dt_grid.best_estimator_,
                'best_params': dt_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': dt_time,
                'predictions_val': y_pred_val
            }
        
        # ====================================================================
        # RANDOM FOREST
        # ====================================================================
        if self.verbose:
            print("\n[2/3] Entrenando Random Forest...")
            print("-" * 80)
        
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],  # Limitar profundidad para evitar overfitting
            'min_samples_split': [10, 20, 50],  # Más regularización
            'min_samples_leaf': [5, 10, 20],  # Más regularización
            'max_features': ['sqrt', 'log2']  # Regularización de features
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
            RandomForestRegressor(random_state=self.random_state),
            param_grid_rf,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        rf_grid.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        self.models['Random Forest'] = rf_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {rf_time:.2f}s")
            print(f"  Mejores parámetros: {rf_grid.best_params_}")
            print(f"  Mejor CV MAE: {-rf_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val is not None and y_val is not None:
            y_pred_val = rf_grid.predict(X_val)
            y_pred_train = rf_grid.predict(X_train)
            
            self.results['Random Forest'] = {
                'model': rf_grid.best_estimator_,
                'best_params': rf_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': rf_time,
                'predictions_val': y_pred_val,
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_grid.best_estimator_.feature_importances_
                }).sort_values('importance', ascending=False)
            }
        
        # ====================================================================
        # EXTRA TREES (usa parámetros de Random Forest)
        # ====================================================================
        if self.verbose:
            print("\n[3/3] Entrenando Extra Trees...")
            print("-" * 80)
            print("  Usando parámetros similares a Random Forest")
        
        best_rf_params = rf_grid.best_params_
        
        start_time = time.time()
        
        et_model = ExtraTreesRegressor(
            n_estimators=best_rf_params['n_estimators'],
            max_depth=best_rf_params['max_depth'],
            min_samples_split=best_rf_params['min_samples_split'],
            random_state=self.random_state,
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
            
            self.results['Extra Trees'] = {
                'model': et_model,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': et_time,
                'predictions_val': y_pred_val
            }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA ÁRBOLES".center(80))
            print(f"{'=' * 80}")
            for model_name in ['Decision Tree', 'Random Forest', 'Extra Trees']:
                if model_name in self.results:
                    r = self.results[model_name]
                    print(f"\n  {model_name}:")
                    print(f"    MAE (val):   {r['mae_val']:.4f}")
                    print(f"    MAE (train): {r['mae_train']:.4f}")
                    print(f"    Tiempo:      {r['train_time']:.2f}s")
        
        return self
    
    def predict(self, X, model_name='Random Forest'):
        """Predice usando el modelo especificado"""
        return self.models[model_name].predict(X)
    
    def get_results(self):
        """Retorna diccionario con todos los resultados"""
        return self.results


# ============================================================================
# FAMILIA BOOSTING (Gradient Boosting, XGBoost, LightGBM)
# ============================================================================

class BoostingFamily(BaseEstimator):
    """
    Familia de métodos de ensamble basados en boosting.
    
    Usa MinMaxScaler para preprocesamiento. Incluye:
    - Gradient Boosting (sklearn) con GridSearchCV
    - XGBoost con early stopping
    - LightGBM optimizado para velocidad
    """
    
    def __init__(self, cv_folds=3, random_state=42, n_jobs=-1, verbose=True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena todos los modelos de boosting"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FAMILIA BOOSTING: GRADIENT BOOSTING, XGBOOST, LIGHTGBM".center(80))
            print("=" * 80)
        
        # ====================================================================
        # PREPROCESAMIENTO: MinMaxScaler
        # ====================================================================
        if self.verbose:
            print("\n[0/3] Aplicando MinMaxScaler...")
        
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        if self.verbose:
            print(f"  ✓ Datos escalados: rango=[{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
        
        # ====================================================================
        # GRADIENT BOOSTING
        # ====================================================================
        if self.verbose:
            print("\n[1/3] Entrenando Gradient Boosting...")
            print("-" * 80)
        
        param_grid_gb = {
            'learning_rate': [0.01, 0.05],  # Reducir opciones
            'n_estimators': [50, 100, 150],  # Reducir para evitar overfitting
            'max_depth': [2, 3, 4],  # Reducir profundidad
            'subsample': [0.7, 0.8],  # Añadir regularización
            'min_samples_split': [10, 20],  # Añadir regularización
            'min_samples_leaf': [5, 10]  # Añadir regularización
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
            GradientBoostingRegressor(random_state=self.random_state),
            param_grid_gb,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        gb_grid.fit(X_train_scaled, y_train)
        gb_time = time.time() - start_time
        
        self.models['Gradient Boosting'] = gb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {gb_time:.2f}s")
            print(f"  Mejores parámetros: {gb_grid.best_params_}")
            print(f"  Mejor CV MAE: {-gb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
            y_pred_val = gb_grid.predict(X_val_scaled)
            y_pred_train = gb_grid.predict(X_train_scaled)
            
            self.results['Gradient Boosting'] = {
                'model': gb_grid.best_estimator_,
                'best_params': gb_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': gb_time,
                'predictions_val': y_pred_val
            }
        
        # ====================================================================
        # XGBOOST CON GRIDSEARCH
        # ====================================================================
        if self.verbose:
            print("\n[2/3] Entrenando XGBoost...")
            print("-" * 80)
        
        param_grid_xgb = {
            'n_estimators': [100, 200],  # Pocas opciones para evitar overfitting
            'learning_rate': [0.01, 0.05],  # Learning rates conservadores
            'max_depth': [3, 4],  # Profundidad limitada
            'subsample': [0.7, 0.8],  # Regularización
            'colsample_bytree': [0.7, 0.8],  # Regularización
            'reg_alpha': [0.5, 1.0],  # Regularización L1
            'reg_lambda': [0.5, 1.0]  # Regularización L2
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
            XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
            param_grid_xgb,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        xgb_grid.fit(X_train_scaled, y_train)
        xgb_time = time.time() - start_time
        
        self.models['XGBoost'] = xgb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {xgb_time:.2f}s")
            print(f"  Mejores parámetros: {xgb_grid.best_params_}")
            print(f"  Mejor CV MAE: {-xgb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
            y_pred_val = xgb_grid.predict(X_val_scaled)
            y_pred_train = xgb_grid.predict(X_train_scaled)
            
            self.results['XGBoost'] = {
                'model': xgb_grid.best_estimator_,
                'best_params': xgb_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': xgb_time,
                'predictions_val': y_pred_val
            }
        
        # ====================================================================
        # LIGHTGBM CON GRIDSEARCH
        # ====================================================================
        if self.verbose:
            print("\n[3/3] Entrenando LightGBM...")
            print("-" * 80)
        
        param_grid_lgb = {
            'n_estimators': [100, 200],  # Pocas opciones para evitar overfitting
            'learning_rate': [0.01, 0.05],  # Learning rates conservadores
            'max_depth': [3, 4],  # Profundidad limitada
            'subsample': [0.7, 0.8],  # Regularización
            'colsample_bytree': [0.7, 0.8],  # Regularización
            'reg_alpha': [0.5, 1.0],  # Regularización L1
            'reg_lambda': [0.5, 1.0],  # Regularización L2
            'min_child_samples': [10, 20]  # Regularización
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
            LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1),
            param_grid_lgb,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        lgb_grid.fit(X_train_scaled, y_train)
        lgb_time = time.time() - start_time
        
        self.models['LightGBM'] = lgb_grid.best_estimator_
        
        if self.verbose:
            print(f"  ✓ Completado en {lgb_time:.2f}s")
            print(f"  Mejores parámetros: {lgb_grid.best_params_}")
            print(f"  Mejor CV MAE: {-lgb_grid.best_score_:.4f}")
        
        # Evaluar
        if X_val_scaled is not None and y_val is not None:
            y_pred_val = lgb_grid.predict(X_val_scaled)
            y_pred_train = lgb_grid.predict(X_train_scaled)
            
            self.results['LightGBM'] = {
                'model': lgb_grid.best_estimator_,
                'best_params': lgb_grid.best_params_,
                'mae_val': mean_absolute_error(y_val, y_pred_val),
                'rmse_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'r2_val': r2_score(y_val, y_pred_val),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'train_time': lgb_time,
                'predictions_val': y_pred_val
            }
        
        # Resumen
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("RESUMEN FAMILIA BOOSTING".center(80))
            print(f"{'=' * 80}")
            for model_name in ['Gradient Boosting', 'XGBoost', 'LightGBM']:
                if model_name in self.results:
                    r = self.results[model_name]
                    print(f"\n  {model_name}:")
                    print(f"    MAE (val):   {r['mae_val']:.4f}")
                    print(f"    MAE (train): {r['mae_train']:.4f}")
                    print(f"    Tiempo:      {r['train_time']:.2f}s")
        
        return self
    
    def predict(self, X, model_name='LightGBM'):
        """Predice usando el modelo especificado"""
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)
    
    def get_results(self):
        """Retorna diccionario con todos los resultados"""
        return self.results


# ============================================================================
# ORQUESTADOR PRINCIPAL - PIPELINE DE ENTRENAMIENTO COMPLETO
# ============================================================================
# ============================================================================
# ORQUESTADOR PRINCIPAL - PIPELINE DE ENTRENAMIENTO COMPLETO
# ============================================================================

class ModelTrainingPipeline(BaseEstimator):
    """
    Orquestador principal que coordina el entrenamiento de todas las familias.
    
    Ejecuta en secuencia:
    1. Familia SVM (con búsqueda optimizada en 2 fases)
    2. Familia Árboles (Decision Tree, Random Forest, Extra Trees)
    3. Familia Boosting (Gradient Boosting, XGBoost, LightGBM)
    
    Genera comparación automática y selecciona el mejor modelo.
    """
    
    def __init__(self, cv_folds=3, random_state=42, n_jobs=-1, verbose=True, 
                 families=['svm', 'trees', 'boosting']):
        """
        Args:
            cv_folds: Número de folds para cross-validation
            random_state: Semilla para reproducibilidad
            n_jobs: Número de cores (-1 usa todos)
            verbose: Mostrar información detallada
            families: Lista de familias a entrenar. Opciones: 'svm', 'trees', 'boosting'
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.families_to_train = [f.lower() for f in families]
        
        # Familias de modelos
        self.svm_family = None
        self.tree_family = None
        self.boosting_family = None
        
        # Resultados
        self.all_results = {}
        self.comparison_df = None
        self.best_model_info = None
        self.total_time = 0
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena todas las familias de modelos
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
        
        Returns:
            self: Instancia del pipeline entrenado
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(" PIPELINE DE ENTRENAMIENTO - INICIO ".center(80, "="))
            print("=" * 80)
            print(f"\nDimensiones de datos:")
            print(f"  Train: {X_train.shape}")
            if X_val is not None:
                print(f"  Val:   {X_val.shape}")
            print(f"  CV Folds: {self.cv_folds}")
            print(f"  Random State: {self.random_state}")
            print(f"  Familias a entrenar: {[f.upper() for f in self.families_to_train]}")
        
        start_total = time.time()
        
        family_count = 0
        total_families = len(self.families_to_train)
        
        # ====================================================================
        # FAMILIA 1: SVM (OPCIONAL)
        # ====================================================================
        if 'svm' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA SVM ".center(80, "="))
                print("=" * 80)
            
            self.svm_family = SVMFamily(
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            self.svm_family.fit(X_train, y_train, X_val, y_val)
            self.all_results['SVM'] = self.svm_family.get_results()
        
        # ====================================================================
        # FAMILIA 2: ÁRBOLES (OPCIONAL)
        # ====================================================================
        if 'trees' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA ÁRBOLES ".center(80, "="))
                print("=" * 80)
            
            self.tree_family = TreeFamily(
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            self.tree_family.fit(X_train, y_train, X_val, y_val)
            self.all_results['Trees'] = self.tree_family.get_results()
        
        # ====================================================================
        # FAMILIA 3: BOOSTING (OPCIONAL)
        # ====================================================================
        if 'boosting' in self.families_to_train:
            family_count += 1
            if self.verbose:
                print("\n" + "=" * 80)
                print(f" [{family_count}/{total_families}] ENTRENANDO FAMILIA BOOSTING ".center(80, "="))
                print("=" * 80)
            
            self.boosting_family = BoostingFamily(
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            self.boosting_family.fit(X_train, y_train, X_val, y_val)
            self.all_results['Boosting'] = self.boosting_family.get_results()
        
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
        
        # SVM
        if 'SVM' in self.all_results:
            comparison_data.append({
                'Familia': 'SVM',
                'Modelo': 'SVM',
                'MAE_val': self.all_results['SVM']['mae_val'],
                'RMSE_val': self.all_results['SVM']['rmse_val'],
                'R2_val': self.all_results['SVM']['r2_val'],
                'MAE_train': self.all_results['SVM']['mae_train'],
                'Overfitting': abs(self.all_results['SVM']['mae_train'] - 
                                  self.all_results['SVM']['mae_val']),
                'Tiempo_seg': self.all_results['SVM']['train_time']
            })
        
        # Árboles
        if 'Trees' in self.all_results:
            for model_name in ['Decision Tree', 'Random Forest', 'Extra Trees']:
                if model_name in self.all_results['Trees']:
                    r = self.all_results['Trees'][model_name]
                    comparison_data.append({
                        'Familia': 'Árboles',
                        'Modelo': model_name,
                        'MAE_val': r['mae_val'],
                        'RMSE_val': r['rmse_val'],
                        'R2_val': r['r2_val'],
                        'MAE_train': r['mae_train'],
                        'Overfitting': abs(r['mae_train'] - r['mae_val']),
                        'Tiempo_seg': r['train_time']
                    })
        
        # Boosting
        if 'Boosting' in self.all_results:
            for model_name in ['Gradient Boosting', 'XGBoost', 'LightGBM']:
                if model_name in self.all_results['Boosting']:
                    r = self.all_results['Boosting'][model_name]
                    comparison_data.append({
                        'Familia': 'Boosting',
                        'Modelo': model_name,
                        'MAE_val': r['mae_val'],
                        'RMSE_val': r['rmse_val'],
                        'R2_val': r['r2_val'],
                        'MAE_train': r['mae_train'],
                        'Overfitting': abs(r['mae_train'] - r['mae_val']),
                        'Tiempo_seg': r['train_time']
                    })
        
        # Crear DataFrame y ordenar por MAE
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('MAE_val').reset_index(drop=True)
        
        # Identificar mejor modelo
        best_idx = self.comparison_df['MAE_val'].idxmin()
        self.best_model_info = self.comparison_df.loc[best_idx].to_dict()
    
    def _print_summary(self):
        """Imprime resumen comparativo"""
        print("\n" + "=" * 80)
        print(" RESUMEN COMPARATIVO - TODOS LOS MODELOS ".center(80, "="))
        print("=" * 80)
        
        print("\nRanking por MAE en validación:")
        print(self.comparison_df.to_string(index=False))
        
        print(f"\n{'=' * 80}")
        print(f"Tiempo total de entrenamiento: {self.total_time:.2f}s ({self.total_time/60:.2f} min)")
        print(f"{'=' * 80}")
        
        print(f"\n🏆 MEJOR MODELO:")
        print(f"   Familia:      {self.best_model_info['Familia']}")
        print(f"   Modelo:       {self.best_model_info['Modelo']}")
        print(f"   MAE (val):    {self.best_model_info['MAE_val']:.4f}")
        print(f"   RMSE (val):   {self.best_model_info['RMSE_val']:.4f}")
        print(f"   R² (val):     {self.best_model_info['R2_val']:.4f}")
        print(f"   Overfitting:  {self.best_model_info['Overfitting']:.4f}")
        print(f"   Tiempo:       {self.best_model_info['Tiempo_seg']:.2f}s")
    
    def predict(self, X, family='best', model_name=None):
        """
        Realiza predicciones con el modelo especificado
        
        Args:
            X: Features para predecir
            family: 'best', 'svm', 'trees', o 'boosting'
            model_name: Nombre específico del modelo (ej: 'Random Forest')
        
        Returns:
            np.array: Predicciones
        """
        if family == 'best':
            family = self.best_model_info['Familia'].lower()
            model_name = self.best_model_info['Modelo']
        
        if 'svm' in family.lower():
            return self.svm_family.predict(X)
        elif 'árbol' in family.lower() or 'tree' in family.lower():
            return self.tree_family.predict(X, model_name=model_name)
        elif 'boost' in family.lower():
            return self.boosting_family.predict(X, model_name=model_name)
        else:
            raise ValueError(f"Familia desconocida: {family}")
    
    def get_best_model(self):
        """
        Retorna el mejor modelo entrenado
        
        Returns:
            tuple: (modelo, información del modelo)
        """
        family = self.best_model_info['Familia']
        model_name = self.best_model_info['Modelo']
        
        if family == 'SVM':
            return self.all_results['SVM']['model'], self.best_model_info
        elif family == 'Árboles':
            return self.all_results['Trees'][model_name]['model'], self.best_model_info
        elif family == 'Boosting':
            return self.all_results['Boosting'][model_name]['model'], self.best_model_info
    
    def get_comparison_df(self):
        """Retorna DataFrame con comparación de todos los modelos"""
        return self.comparison_df
    
    def get_all_results(self):
        """Retorna diccionario con todos los resultados detallados"""
        return self.all_results
    
    def save_results(self, output_dir='.'):
        """
        Guarda todos los resultados en archivos
        
        Args:
            output_dir: Directorio donde guardar los archivos
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Guardar comparación
        comparison_path = output_dir / 'model_comparison.csv'
        self.comparison_df.to_csv(comparison_path, index=False)
        
        # Guardar resumen en texto
        summary_path = output_dir / 'training_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" RESUMEN DE ENTRENAMIENTO ".center(80) + "\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPARACIÓN DE MODELOS:\n")
            f.write(self.comparison_df.to_string(index=False) + "\n\n")
            
            f.write("MEJOR MODELO:\n")
            for key, value in self.best_model_info.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTiempo total: {self.total_time:.2f}s ({self.total_time/60:.2f} min)\n")
        
        if self.verbose:
            print(f"\n✓ Resultados guardados en: {output_dir}")
            print(f"  - {comparison_path.name}")
            print(f"  - {summary_path.name}")
    
    def plot_comparison(self, metric='MAE_val', figsize=(12, 6)):
        """
        Visualiza comparación de modelos
        
        Args:
            metric: Métrica a visualizar ('MAE_val', 'RMSE_val', 'R2_val')
            figsize: Tamaño de la figura
        """
        plt.figure(figsize=figsize)
        
        colors = {'SVM': '#FF6B6B', 'Árboles': '#4ECDC4', 'Boosting': '#45B7D1'}
        df = self.comparison_df.copy()
        df['Color'] = df['Familia'].map(colors)
        
        bars = plt.barh(df['Modelo'], df[metric], color=df['Color'])
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Modelo')
        plt.title(f'Comparación de Modelos - {metric.replace("_", " ").title()}')
        plt.grid(axis='x', alpha=0.3)
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=familia) 
                          for familia, color in colors.items()]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n=20):
        """
        Retorna feature importance de Random Forest (si está disponible)
        
        Args:
            top_n: Número de features más importantes a mostrar
        
        Returns:
            pd.DataFrame: Features ordenadas por importancia
        """
        if 'Trees' in self.all_results and 'Random Forest' in self.all_results['Trees']:
            rf_results = self.all_results['Trees']['Random Forest']
            if 'feature_importance' in rf_results:
                return rf_results['feature_importance'].head(top_n)
        
        return None