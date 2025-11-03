"""
Sentiment Features Transformer for Google Play Store Apps

Este módulo agrega características derivadas del análisis de sentimientos de reviews
para mejorar la predicción de ratings.

Features generadas:
- Distribución de sentimientos (positive/negative/neutral ratios)
- Estadísticas de polaridad (mean, median, std, min, max)
- Estadísticas de subjetividad (mean, median)
- Métricas de volumen (total reviews con sentiment, ratio)
- Sentiment strength (fuerza de sentimientos positivos/negativos)

Autor: Sistema de ML
Fecha: 2025
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae features de sentiment analysis a nivel de app agregando reviews.
    
    Para cada app calcula:
    - Distribución de sentimientos (positive/negative/neutral)
    - Estadísticas de polaridad y subjetividad
    - Métricas de volumen de reviews
    - Sentiment strength
    
    Ejemplo de uso:
        >>> extractor = SentimentFeatureExtractor('reviews.csv', verbose=True)
        >>> df_with_sentiment = extractor.fit_transform(df_apps)
    """
    
    def __init__(self, reviews_path: str, verbose: bool = True):
        """
        Args:
            reviews_path: Ruta al archivo CSV de reviews
            verbose: Mostrar información detallada
        """
        self.reviews_path = reviews_path
        self.verbose = verbose
        self.sentiment_features = None
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """
        Calcula features de sentiment por app desde el archivo de reviews.
        
        Args:
            X: DataFrame con columna 'App'
            y: Target (no utilizado)
            
        Returns:
            self
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(" EXTRAYENDO FEATURES DE SENTIMENT ANALYSIS ".center(80, "="))
            print("=" * 80)
        
        # Cargar reviews
        try:
            reviews_df = pd.read_csv(self.reviews_path)
            if self.verbose:
                print(f"\n✓ Reviews cargados: {len(reviews_df):,} filas")
                print(f"  Columnas: {list(reviews_df.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo de reviews no encontrado: {self.reviews_path}")
        
        # Validar columnas requeridas
        required_cols = ['App', 'Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
        missing_cols = [col for col in required_cols if col not in reviews_df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en reviews: {missing_cols}")
        
        # Filtrar reviews válidos (no NaN en sentiment)
        reviews_valid = reviews_df.dropna(subset=['Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity'])
        
        if self.verbose:
            print(f"\n✓ Reviews válidos (con sentiment): {len(reviews_valid):,} ({len(reviews_valid)/len(reviews_df)*100:.1f}%)")
            print(f"  Removidos (NaN): {len(reviews_df) - len(reviews_valid):,}")
        
        # Agrupar por App y calcular features
        if self.verbose:
            print("\nCalculando features de sentiment por app...")
            print("-" * 80)
        
        sentiment_agg = []
        
        for app_name, group in reviews_valid.groupby('App'):
            features = {'App': app_name}
            
            # ================================================================
            # 1. DISTRIBUCIÓN DE SENTIMIENTOS (3 features)
            # ================================================================
            sentiment_counts = group['Sentiment'].value_counts(normalize=True)
            features['positive_ratio'] = sentiment_counts.get('Positive', 0)
            features['negative_ratio'] = sentiment_counts.get('Negative', 0)
            features['neutral_ratio'] = sentiment_counts.get('Neutral', 0)
            
            # ================================================================
            # 2. ESTADÍSTICAS DE POLARIDAD (6 features)
            # ================================================================
            polarity = group['Sentiment_Polarity']
            features['polarity_mean'] = polarity.mean()
            features['polarity_median'] = polarity.median()
            features['polarity_std'] = polarity.std()
            features['polarity_min'] = polarity.min()
            features['polarity_max'] = polarity.max()
            features['polarity_range'] = polarity.max() - polarity.min()
            
            # ================================================================
            # 3. SUBJETIVIDAD (3 features)
            # ================================================================
            subjectivity = group['Sentiment_Subjectivity']
            features['subjectivity_mean'] = subjectivity.mean()
            features['subjectivity_median'] = subjectivity.median()
            features['subjectivity_std'] = subjectivity.std()
            
            # ================================================================
            # 4. SENTIMENT STRENGTH (2 features)
            # ================================================================
            positive_reviews = group[group['Sentiment'] == 'Positive']
            features['positive_strength'] = positive_reviews['Sentiment_Polarity'].mean() if len(positive_reviews) > 0 else 0
            
            negative_reviews = group[group['Sentiment'] == 'Negative']
            features['negative_strength'] = abs(negative_reviews['Sentiment_Polarity'].mean()) if len(negative_reviews) > 0 else 0
            
            # ================================================================
            # 5. FEATURES ADICIONALES (3 features)
            # ================================================================
            features['total_reviews_sentiment'] = len(group)
            features['opinion_variance'] = polarity.var()
            features['extreme_polarity_ratio'] = ((abs(polarity) > 0.5).sum() / len(polarity)) if len(polarity) > 0 else 0
            
            sentiment_agg.append(features)
        
        # Crear DataFrame con features de sentiment
        self.sentiment_features = pd.DataFrame(sentiment_agg)
        
        if self.verbose:
            print(f"\n✓ Features de sentiment calculadas para {len(self.sentiment_features):,} apps")
            print(f"  Features por app: {len(self.sentiment_features.columns) - 1}")
            print(f"\nEjemplo de features calculadas (primeras 3 apps):")
            display_cols = ['App', 'positive_ratio', 'negative_ratio', 'polarity_mean', 
                           'polarity_std', 'negative_strength', 'total_reviews_sentiment']
            print(self.sentiment_features[display_cols].head(3).to_string(index=False))
            
            # Estadísticas de las features
            print(f"\n📊 Estadísticas de sentiment features:")
            print(f"  Total features: {len(self.sentiment_features.columns) - 1} (18 features completas)")
            print(f"  Positive ratio: mean={self.sentiment_features['positive_ratio'].mean():.3f}")
            print(f"  Negative ratio: mean={self.sentiment_features['negative_ratio'].mean():.3f}")
            print(f"  Polarity mean: mean={self.sentiment_features['polarity_mean'].mean():.3f}, std={self.sentiment_features['polarity_mean'].std():.3f}")
            print(f"  Total reviews per app: mean={self.sentiment_features['total_reviews_sentiment'].mean():.1f}")
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Hace merge de features de sentiment con el DataFrame de apps.
        
        Args:
            X: DataFrame con columna 'App'
            
        Returns:
            DataFrame con features de sentiment agregadas
        """
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() antes de transform()")
        
        df = X.copy()
        
        # Merge con sentiment features
        df = df.merge(self.sentiment_features, on='App', how='left')
        
        # Contar apps sin sentiment data
        apps_without_sentiment = df[self.sentiment_features.columns[1]].isna().sum()
        
        if self.verbose and apps_without_sentiment > 0:
            print(f"\n⚠️  Apps sin reviews de sentiment: {apps_without_sentiment} ({apps_without_sentiment/len(df)*100:.1f}%)")
            print(f"   → Se imputarán con valores por defecto")
        
        # Imputar apps sin sentiment con valores conservadores
        sentiment_cols = self.sentiment_features.columns.drop('App').tolist()
        
        # Valores por defecto (representan "sin información de sentiment")
        default_values = {
            # Distribución neutra (3 features)
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            # Polaridad neutra (6 features)
            'polarity_mean': 0.0,
            'polarity_median': 0.0,
            'polarity_std': 0.3,
            'polarity_min': -0.5,
            'polarity_max': 0.5,
            'polarity_range': 1.0,
            # Subjetividad moderada (3 features)
            'subjectivity_mean': 0.5,
            'subjectivity_median': 0.5,
            'subjectivity_std': 0.3,
            # Sin fuerza de sentimientos (2 features)
            'positive_strength': 0.0,
            'negative_strength': 0.0,
            # Características adicionales (3 features)
            'total_reviews_sentiment': 0,
            'opinion_variance': 0.09,
            'extreme_polarity_ratio': 0.0
        }
        
        for col in sentiment_cols:
            if col in default_values:
                df[col] = df[col].fillna(default_values[col])
        
        # Crear feature adicional: ratio de reviews con sentiment vs Reviews totales
        if 'Reviews' in df.columns:
            df['reviews_with_sentiment_ratio'] = df['total_reviews_sentiment'] / (df['Reviews'] + 1)
            if self.verbose:
                print(f"\n✓ Feature adicional creada: reviews_with_sentiment_ratio")
                print(f"  Mean: {df['reviews_with_sentiment_ratio'].mean():.4f}")
        
        if self.verbose:
            print(f"\n✓ Sentiment features agregadas al DataFrame")
            print(f"  Shape final: {df.shape}")
            print(f"  Features de sentiment: {len(sentiment_cols) + 1}")  # +1 por reviews_with_sentiment_ratio
        
        return df
    
    def fit_transform(self, X, y=None):
        """Fit y transform en un solo paso"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self):
        """Retorna lista de nombres de features generadas"""
        if not self._is_fitted:
            raise ValueError("Debe llamar fit() primero")
        
        feature_names = self.sentiment_features.columns.drop('App').tolist()
        feature_names.append('reviews_with_sentiment_ratio')
        return feature_names
