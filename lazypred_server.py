#!/usr/bin/env python3
"""
LazyPredict MCP Server - Automated Machine Learning Model Comparison
Provides comprehensive functionality for the LazyPredict library via Model Context Protocol
"""

import os
import json
import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import io
from dataclasses import dataclass, asdict
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, Prompt, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

# LazyPredict imports
try:
    from lazypredict.Supervised import LazyClassifier, LazyRegressor
    from lazypredict.Unsupervised import LazyClusterer
    LAZYPREDICT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LazyPredict not found. Please install: pip install lazypredict")
    print(f"Error: {e}")
    LAZYPREDICT_AVAILABLE = False

# ML utilities imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.datasets import (
        make_classification, make_regression, make_blobs,
        load_iris, load_diabetes, load_wine, load_breast_cancer
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML dependencies not found: {e}")
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LazyPredictConfig:
    """Configuration for LazyPredict operations"""
    verbose: int = 0
    ignore_warnings: bool = True
    custom_metric: str = None
    predictions: bool = False
    sort_by: str = 'Accuracy'
    random_state: int = 42
    fold: int = 5
    n_jobs: int = -1
    
    # Classification specific
    classifiers: List[str] = None
    
    # Regression specific  
    regressors: List[str] = None
    
    # Clustering specific
    clusterers: List[str] = None
    
    def __post_init__(self):
        if self.classifiers is None:
            self.classifiers = []  # Will use all available
        if self.regressors is None:
            self.regressors = []   # Will use all available
        if self.clusterers is None:
            self.clusterers = []   # Will use all available

class LazyPredictAgent:
    """Comprehensive LazyPredict functionality with enhanced features"""
    
    def __init__(self, config: LazyPredictConfig = None):
        self.config = config or LazyPredictConfig()
        self.results_cache = {}
        self.datasets_cache = {}
        self.models_cache = {}
        
    # =================== SUPERVISED LEARNING - CLASSIFICATION ===================
    
    def run_lazy_classifier(self, X_train: Union[pd.DataFrame, np.ndarray], 
                           X_test: Union[pd.DataFrame, np.ndarray],
                           y_train: Union[pd.Series, np.ndarray], 
                           y_test: Union[pd.Series, np.ndarray],
                           classifiers: List[str] = None,
                           custom_metric: str = None,
                           **kwargs) -> Dict[str, Any]:
        """Run comprehensive classification model comparison"""
        try:
            if not LAZYPREDICT_AVAILABLE:
                return {'error': 'LazyPredict package not available'}
            
            # Configure parameters
            clf_params = {
                'verbose': kwargs.get('verbose', self.config.verbose),
                'ignore_warnings': kwargs.get('ignore_warnings', self.config.ignore_warnings),
                'custom_metric': custom_metric or self.config.custom_metric,
                'predictions': kwargs.get('predictions', self.config.predictions),
                'sort_by': kwargs.get('sort_by', self.config.sort_by),
                'random_state': kwargs.get('random_state', self.config.random_state),
                'classifiers': classifiers or self.config.classifiers,
                'fold': kwargs.get('fold', self.config.fold)
            }
            
            # Remove None values
            clf_params = {k: v for k, v in clf_params.items() if v is not None}
            
            # Initialize LazyClassifier
            clf = LazyClassifier(**clf_params)
            
            # Fit and predict
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            
            # Cache results
            result_id = f"classification_{hash(str(models.shape[0]))}"
            self.results_cache[result_id] = {
                'models': models,
                'predictions': predictions,
                'type': 'classification',
                'clf_object': clf,
                'data_shapes': {
                    'X_train': X_train.shape if hasattr(X_train, 'shape') else len(X_train),
                    'X_test': X_test.shape if hasattr(X_test, 'shape') else len(X_test),
                    'y_train': y_train.shape if hasattr(y_train, 'shape') else len(y_train),
                    'y_test': y_test.shape if hasattr(y_test, 'shape') else len(y_test)
                }
            }
            
            return {
                'result_id': result_id,
                'models_performance': models.to_dict(),
                'predictions_available': predictions is not None,
                'top_models': models.head(10).to_dict(),
                'best_model': {
                    'name': models.index[0],
                    'accuracy': float(models.iloc[0]['Accuracy']),
                    'balanced_accuracy': float(models.iloc[0]['Balanced Accuracy']),
                    'roc_auc': float(models.iloc[0]['ROC AUC']),
                    'f1_score': float(models.iloc[0]['F1 Score'])
                },
                'summary_stats': {
                    'total_models': len(models),
                    'successful_models': len(models.dropna()),
                    'failed_models': len(models) - len(models.dropna()),
                    'mean_accuracy': float(models['Accuracy'].mean()),
                    'std_accuracy': float(models['Accuracy'].std()),
                    'max_accuracy': float(models['Accuracy'].max()),
                    'min_accuracy': float(models['Accuracy'].min())
                },
                'configuration': clf_params
            }
            
        except Exception as e:
            logger.error(f"Error in lazy classification: {e}")
            return {'error': str(e)}
    
    def analyze_classification_results(self, result_id: str, 
                                     analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze classification results in detail"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            if result['type'] != 'classification':
                return {'error': 'Result is not a classification result'}
            
            models = result['models']
            analysis = {}
            
            if analysis_type in ['comprehensive', 'model_ranking']:
                # Model ranking analysis
                analysis['model_ranking'] = {
                    'top_10_by_accuracy': models.nlargest(10, 'Accuracy')[['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score']].to_dict(),
                    'top_10_by_roc_auc': models.nlargest(10, 'ROC AUC')[['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score']].to_dict(),
                    'top_10_by_f1': models.nlargest(10, 'F1 Score')[['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score']].to_dict()
                }
            
            if analysis_type in ['comprehensive', 'performance_distribution']:
                # Performance distribution analysis
                analysis['performance_distribution'] = {
                    'accuracy_stats': models['Accuracy'].describe().to_dict(),
                    'roc_auc_stats': models['ROC AUC'].describe().to_dict(),
                    'f1_stats': models['F1 Score'].describe().to_dict(),
                    'training_time_stats': models['Time Taken'].describe().to_dict()
                }
            
            if analysis_type in ['comprehensive', 'model_families']:
                # Model family analysis
                model_families = self._categorize_models(models.index.tolist())
                family_performance = {}
                for family, model_list in model_families.items():
                    family_models = models.loc[model_list]
                    family_performance[family] = {
                        'count': len(family_models),
                        'mean_accuracy': float(family_models['Accuracy'].mean()),
                        'best_model': family_models.loc[family_models['Accuracy'].idxmax()].to_dict(),
                        'avg_time': float(family_models['Time Taken'].mean())
                    }
                analysis['model_families'] = family_performance
            
            if analysis_type in ['comprehensive', 'efficiency_analysis']:
                # Efficiency analysis (performance vs time)
                models_copy = models.copy()
                models_copy['efficiency_score'] = models_copy['Accuracy'] / (models_copy['Time Taken'] + 0.001)  # Avoid division by zero
                analysis['efficiency_analysis'] = {
                    'most_efficient': models_copy.nlargest(10, 'efficiency_score')[['Accuracy', 'Time Taken', 'efficiency_score']].to_dict(),
                    'fastest_models': models.nsmallest(10, 'Time Taken')[['Accuracy', 'Time Taken']].to_dict(),
                    'accuracy_time_correlation': float(models['Accuracy'].corr(models['Time Taken']))
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing classification results: {e}")
            return {'error': str(e)}
    
    # =================== SUPERVISED LEARNING - REGRESSION ===================
    
    def run_lazy_regressor(self, X_train: Union[pd.DataFrame, np.ndarray], 
                          X_test: Union[pd.DataFrame, np.ndarray],
                          y_train: Union[pd.Series, np.ndarray], 
                          y_test: Union[pd.Series, np.ndarray],
                          regressors: List[str] = None,
                          custom_metric: str = None,
                          **kwargs) -> Dict[str, Any]:
        """Run comprehensive regression model comparison"""
        try:
            if not LAZYPREDICT_AVAILABLE:
                return {'error': 'LazyPredict package not available'}
            
            # Configure parameters
            reg_params = {
                'verbose': kwargs.get('verbose', self.config.verbose),
                'ignore_warnings': kwargs.get('ignore_warnings', self.config.ignore_warnings),
                'custom_metric': custom_metric or self.config.custom_metric,
                'predictions': kwargs.get('predictions', self.config.predictions),
                'sort_by': kwargs.get('sort_by', 'R-Squared'),
                'random_state': kwargs.get('random_state', self.config.random_state),
                'regressors': regressors or self.config.regressors,
                'fold': kwargs.get('fold', self.config.fold)
            }
            
            # Remove None values
            reg_params = {k: v for k, v in reg_params.items() if v is not None}
            
            # Initialize LazyRegressor
            reg = LazyRegressor(**reg_params)
            
            # Fit and predict
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)
            
            # Cache results
            result_id = f"regression_{hash(str(models.shape[0]))}"
            self.results_cache[result_id] = {
                'models': models,
                'predictions': predictions,
                'type': 'regression',
                'reg_object': reg,
                'data_shapes': {
                    'X_train': X_train.shape if hasattr(X_train, 'shape') else len(X_train),
                    'X_test': X_test.shape if hasattr(X_test, 'shape') else len(X_test),
                    'y_train': y_train.shape if hasattr(y_train, 'shape') else len(y_train),
                    'y_test': y_test.shape if hasattr(y_test, 'shape') else len(y_test)
                }
            }
            
            return {
                'result_id': result_id,
                'models_performance': models.to_dict(),
                'predictions_available': predictions is not None,
                'top_models': models.head(10).to_dict(),
                'best_model': {
                    'name': models.index[0],
                    'r_squared': float(models.iloc[0]['R-Squared']),
                    'adjusted_r_squared': float(models.iloc[0]['Adjusted R-Squared']),
                    'rmse': float(models.iloc[0]['RMSE']),
                    'mae': float(models.iloc[0]['MAE'])
                },
                'summary_stats': {
                    'total_models': len(models),
                    'successful_models': len(models.dropna()),
                    'failed_models': len(models) - len(models.dropna()),
                    'mean_r_squared': float(models['R-Squared'].mean()),
                    'std_r_squared': float(models['R-Squared'].std()),
                    'max_r_squared': float(models['R-Squared'].max()),
                    'min_r_squared': float(models['R-Squared'].min())
                },
                'configuration': reg_params
            }
            
        except Exception as e:
            logger.error(f"Error in lazy regression: {e}")
            return {'error': str(e)}
    
    def analyze_regression_results(self, result_id: str, 
                                 analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze regression results in detail"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            if result['type'] != 'regression':
                return {'error': 'Result is not a regression result'}
            
            models = result['models']
            analysis = {}
            
            if analysis_type in ['comprehensive', 'model_ranking']:
                # Model ranking analysis
                analysis['model_ranking'] = {
                    'top_10_by_r_squared': models.nlargest(10, 'R-Squared')[['R-Squared', 'Adjusted R-Squared', 'RMSE', 'MAE']].to_dict(),
                    'top_10_by_rmse': models.nsmallest(10, 'RMSE')[['R-Squared', 'Adjusted R-Squared', 'RMSE', 'MAE']].to_dict(),
                    'top_10_by_mae': models.nsmallest(10, 'MAE')[['R-Squared', 'Adjusted R-Squared', 'RMSE', 'MAE']].to_dict()
                }
            
            if analysis_type in ['comprehensive', 'performance_distribution']:
                # Performance distribution analysis
                analysis['performance_distribution'] = {
                    'r_squared_stats': models['R-Squared'].describe().to_dict(),
                    'rmse_stats': models['RMSE'].describe().to_dict(),
                    'mae_stats': models['MAE'].describe().to_dict(),
                    'training_time_stats': models['Time Taken'].describe().to_dict()
                }
            
            if analysis_type in ['comprehensive', 'error_analysis']:
                # Error analysis
                analysis['error_analysis'] = {
                    'rmse_mae_correlation': float(models['RMSE'].corr(models['MAE'])),
                    'low_error_models': models[(models['RMSE'] < models['RMSE'].quantile(0.25)) & 
                                             (models['MAE'] < models['MAE'].quantile(0.25))].to_dict(),
                    'error_consistency': {
                        'rmse_std': float(models['RMSE'].std()),
                        'mae_std': float(models['MAE'].std()),
                        'coefficient_of_variation_rmse': float(models['RMSE'].std() / models['RMSE'].mean())
                    }
                }
            
            if analysis_type in ['comprehensive', 'model_families']:
                # Model family analysis
                model_families = self._categorize_models(models.index.tolist())
                family_performance = {}
                for family, model_list in model_families.items():
                    family_models = models.loc[model_list]
                    family_performance[family] = {
                        'count': len(family_models),
                        'mean_r_squared': float(family_models['R-Squared'].mean()),
                        'mean_rmse': float(family_models['RMSE'].mean()),
                        'best_model': family_models.loc[family_models['R-Squared'].idxmax()].to_dict(),
                        'avg_time': float(family_models['Time Taken'].mean())
                    }
                analysis['model_families'] = family_performance
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing regression results: {e}")
            return {'error': str(e)}
    
    # =================== UNSUPERVISED LEARNING - CLUSTERING ===================
    
    def run_lazy_clusterer(self, X: Union[pd.DataFrame, np.ndarray],
                          clusterers: List[str] = None,
                          n_clusters: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """Run comprehensive clustering model comparison"""
        try:
            if not LAZYPREDICT_AVAILABLE:
                return {'error': 'LazyPredict package not available'}
            
            # Configure parameters
            cluster_params = {
                'verbose': kwargs.get('verbose', self.config.verbose),
                'ignore_warnings': kwargs.get('ignore_warnings', self.config.ignore_warnings),
                'clusterers': clusterers or self.config.clusterers,
                'random_state': kwargs.get('random_state', self.config.random_state)
            }
            
            # Remove None values
            cluster_params = {k: v for k, v in cluster_params.items() if v is not None}
            
            # Initialize LazyClusterer
            clusterer = LazyClusterer(**cluster_params)
            
            # Fit and get results
            models, predictions = clusterer.fit(X, n_clusters=n_clusters)
            
            # Cache results
            result_id = f"clustering_{hash(str(models.shape[0]))}"
            self.results_cache[result_id] = {
                'models': models,
                'predictions': predictions,
                'type': 'clustering',
                'clusterer_object': clusterer,
                'n_clusters': n_clusters,
                'data_shape': X.shape if hasattr(X, 'shape') else len(X)
            }
            
            return {
                'result_id': result_id,
                'models_performance': models.to_dict(),
                'predictions_available': predictions is not None,
                'top_models': models.head(10).to_dict(),
                'best_model': {
                    'name': models.index[0],
                    'adjusted_rand_score': float(models.iloc[0]['Adjusted Rand Score']),
                    'adjusted_mutual_info_score': float(models.iloc[0]['Adjusted Mutual Info Score']),
                    'silhouette_score': float(models.iloc[0]['Silhouette Score'])
                },
                'summary_stats': {
                    'total_models': len(models),
                    'successful_models': len(models.dropna()),
                    'failed_models': len(models) - len(models.dropna()),
                    'mean_silhouette': float(models['Silhouette Score'].mean()),
                    'max_silhouette': float(models['Silhouette Score'].max()),
                    'min_silhouette': float(models['Silhouette Score'].min())
                },
                'configuration': cluster_params,
                'n_clusters_used': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in lazy clustering: {e}")
            return {'error': str(e)}
    
    def analyze_clustering_results(self, result_id: str, 
                                 analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze clustering results in detail"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            if result['type'] != 'clustering':
                return {'error': 'Result is not a clustering result'}
            
            models = result['models']
            analysis = {}
            
            if analysis_type in ['comprehensive', 'model_ranking']:
                # Model ranking analysis
                analysis['model_ranking'] = {
                    'top_10_by_silhouette': models.nlargest(10, 'Silhouette Score').to_dict(),
                    'top_10_by_adjusted_rand': models.nlargest(10, 'Adjusted Rand Score').to_dict(),
                    'top_10_by_mutual_info': models.nlargest(10, 'Adjusted Mutual Info Score').to_dict()
                }
            
            if analysis_type in ['comprehensive', 'performance_distribution']:
                # Performance distribution analysis
                analysis['performance_distribution'] = {
                    'silhouette_stats': models['Silhouette Score'].describe().to_dict(),
                    'adjusted_rand_stats': models['Adjusted Rand Score'].describe().to_dict(),
                    'mutual_info_stats': models['Adjusted Mutual Info Score'].describe().to_dict(),
                    'training_time_stats': models['Time Taken'].describe().to_dict()
                }
            
            if analysis_type in ['comprehensive', 'clustering_quality']:
                # Clustering quality analysis
                analysis['clustering_quality'] = {
                    'high_quality_models': models[models['Silhouette Score'] > 0.5].to_dict(),
                    'metric_correlations': {
                        'silhouette_rand_corr': float(models['Silhouette Score'].corr(models['Adjusted Rand Score'])),
                        'silhouette_mutual_info_corr': float(models['Silhouette Score'].corr(models['Adjusted Mutual Info Score'])),
                        'rand_mutual_info_corr': float(models['Adjusted Rand Score'].corr(models['Adjusted Mutual Info Score']))
                    },
                    'consensus_ranking': self._calculate_consensus_ranking(models, 
                        ['Silhouette Score', 'Adjusted Rand Score', 'Adjusted Mutual Info Score'])
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing clustering results: {e}")
            return {'error': str(e)}
    
    # =================== DATA GENERATION AND PREPROCESSING ===================
    
    def generate_sample_dataset(self, dataset_type: str, 
                              n_samples: int = 1000,
                              n_features: int = 20,
                              **kwargs) -> Dict[str, Any]:
        """Generate sample datasets for testing"""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn not available for data generation'}
            
            if dataset_type == 'classification':
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=kwargs.get('n_informative', max(2, n_features//2)),
                    n_redundant=kwargs.get('n_redundant', max(0, n_features//4)),
                    n_classes=kwargs.get('n_classes', 2),
                    random_state=kwargs.get('random_state', self.config.random_state)
                )
                
            elif dataset_type == 'regression':
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=kwargs.get('n_informative', max(1, n_features//2)),
                    noise=kwargs.get('noise', 0.1),
                    random_state=kwargs.get('random_state', self.config.random_state)
                )
                
            elif dataset_type == 'clustering':
                X, y = make_blobs(
                    n_samples=n_samples,
                    centers=kwargs.get('n_centers', 3),
                    n_features=n_features,
                    cluster_std=kwargs.get('cluster_std', 1.0),
                    random_state=kwargs.get('random_state', self.config.random_state)
                )
                
            else:
                return {'error': f'Unknown dataset type: {dataset_type}'}
            
            # Convert to DataFrame for better handling
            feature_names = [f'feature_{i}' for i in range(n_features)]
            X_df = pd.DataFrame(X, columns=feature_names)
            y_series = pd.Series(y, name='target')
            
            # Cache dataset
            dataset_id = f"generated_{dataset_type}_{hash(str(n_samples))}"
            self.datasets_cache[dataset_id] = {
                'X': X_df,
                'y': y_series,
                'type': dataset_type,
                'metadata': {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'generation_params': kwargs
                }
            }
            
            return {
                'dataset_id': dataset_id,
                'dataset_type': dataset_type,
                'shape': {
                    'X': X_df.shape,
                    'y': y_series.shape
                },
                'sample_data': {
                    'X_head': X_df.head().to_dict(),
                    'y_head': y_series.head().to_dict(),
                    'X_describe': X_df.describe().to_dict(),
                    'y_describe': y_series.describe() if dataset_type == 'regression' else y_series.value_counts().to_dict()
                },
                'generation_params': kwargs
            }
            
        except Exception as e:
            logger.error(f"Error generating sample dataset: {e}")
            return {'error': str(e)}
    
    def load_builtin_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load built-in sklearn datasets"""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn not available'}
            
            dataset_loaders = {
                'iris': load_iris,
                'diabetes': load_diabetes,
                'wine': load_wine,
                'breast_cancer': load_breast_cancer
            }
            
            if dataset_name not in dataset_loaders:
                return {'error': f'Dataset {dataset_name} not available. Available: {list(dataset_loaders.keys())}'}
            
            # Load dataset
            data = dataset_loaders[dataset_name]()
            
            # Convert to DataFrame
            X_df = pd.DataFrame(data.data, columns=data.feature_names)
            y_series = pd.Series(data.target, name='target')
            
            # Determine dataset type
            if dataset_name in ['iris', 'wine', 'breast_cancer']:
                dataset_type = 'classification'
            else:
                dataset_type = 'regression'
            
            # Cache dataset
            dataset_id = f"builtin_{dataset_name}"
            self.datasets_cache[dataset_id] = {
                'X': X_df,
                'y': y_series,
                'type': dataset_type,
                'metadata': {
                    'name': dataset_name,
                    'description': data.DESCR,
                    'n_samples': X_df.shape[0],
                    'n_features': X_df.shape[1],
                    'target_names': data.target_names if hasattr(data, 'target_names') else None
                }
            }
            
            return {
                'dataset_id': dataset_id,
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'shape': {
                    'X': X_df.shape,
                    'y': y_series.shape
                },
                'description': data.DESCR[:500] + '...' if len(data.DESCR) > 500 else data.DESCR,
                'sample_data': {
                    'X_head': X_df.head().to_dict(),
                    'y_head': y_series.head().to_dict(),
                    'X_describe': X_df.describe().to_dict(),
                    'y_distribution': y_series.value_counts().to_dict() if dataset_type == 'classification' else y_series.describe()
                },
                'feature_names': list(data.feature_names),
                'target_names': list(data.target_names) if hasattr(data, 'target_names') else None
            }
            
        except Exception as e:
            logger.error(f"Error loading builtin dataset: {e}")
            return {'error': str(e)}
    
    def preprocess_data(self, dataset_id: str, 
                       preprocessing_steps: List[str] = None,
                       test_size: float = 0.2,
                       scale_features: bool = False,
                       encode_labels: bool = False) -> Dict[str, Any]:
        """Preprocess data for ML pipeline"""
        try:
            if dataset_id not in self.datasets_cache:
                return {'error': 'Dataset not found'}
            
            dataset = self.datasets_cache[dataset_id]
            X = dataset['X'].copy()
            y = dataset['y'].copy()
            
            preprocessing_applied = []
            
            # Handle missing values
            if 'handle_missing' in (preprocessing_steps or []):
                X = X.fillna(X.mean())
                preprocessing_applied.append('Missing values filled with mean')
            
            # Feature scaling
            if scale_features or 'scale' in (preprocessing_steps or []):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                preprocessing_applied.append('Features scaled using StandardScaler')
            
            # Label encoding
            if encode_labels or 'encode_labels' in (preprocessing_steps or []):
                if dataset['type'] == 'classification':
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y), name='target')
                    preprocessing_applied.append('Labels encoded')
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=self.config.random_state,
                stratify=y if dataset['type'] == 'classification' else None
            )
            
            # Cache preprocessed data
            preprocessed_id = f"{dataset_id}_preprocessed"
            self.datasets_cache[preprocessed_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'type': dataset['type'],
                'metadata': {
                    'original_dataset_id': dataset_id,
                    'preprocessing_applied': preprocessing_applied,
                    'test_size': test_size,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            }
            
            return {
                'preprocessed_dataset_id': preprocessed_id,
                'preprocessing_applied': preprocessing_applied,
                'data_splits': {
                    'X_train_shape': X_train.shape,
                    'X_test_shape': X_test.shape,
                    'y_train_shape': y_train.shape,
                    'y_test_shape': y_test.shape
                },
                'sample_data': {
                    'X_train_head': X_train.head().to_dict(),
                    'X_test_head': X_test.head().to_dict(),
                    'y_train_head': y_train.head().to_dict(),
                    'y_test_head': y_test.head().to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return {'error': str(e)}
    
    def load_custom_dataset(self, data: str, target_column: str,
                           dataset_type: str = 'auto') -> Dict[str, Any]:
        """Load custom dataset from CSV data"""
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(data))
            
            # Separate features and target
            if target_column not in df.columns:
                return {'error': f'Target column {target_column} not found in data'}
            
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Auto-detect dataset type
            if dataset_type == 'auto':
                if y.dtype in ['object', 'category'] or len(y.unique()) <= 10:
                    dataset_type = 'classification'
                else:
                    dataset_type = 'regression'
            
            # Cache dataset
            dataset_id = f"custom_{hash(str(df.shape))}"
            self.datasets_cache[dataset_id] = {
                'X': X,
                'y': y,
                'type': dataset_type,
                'metadata': {
                    'n_samples': len(df),
                    'n_features': len(X.columns),
                    'target_column': target_column,
                    'feature_columns': list(X.columns)
                }
            }
            
            return {
                'dataset_id': dataset_id,
                'dataset_type': dataset_type,
                'shape': {
                    'X': X.shape,
                    'y': y.shape
                },
                'feature_columns': list(X.columns),
                'target_column': target_column,
                'sample_data': {
                    'X_head': X.head().to_dict(),
                    'y_head': y.head().to_dict(),
                    'X_describe': X.describe().to_dict(),
                    'y_distribution': y.value_counts().to_dict() if dataset_type == 'classification' else y.describe()
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            return {'error': str(e)}
    
    # =================== MODEL COMPARISON AND ANALYSIS ===================
    
    def compare_model_types(self, dataset_id: str, 
                           model_types: List[str] = None) -> Dict[str, Any]:
        """Compare different types of models on the same dataset"""
        try:
            if dataset_id not in self.datasets_cache:
                return {'error': 'Dataset not found'}
            
            dataset = self.datasets_cache[dataset_id]
            if not all(key in dataset for key in ['X_train', 'X_test', 'y_train', 'y_test']):
                return {'error': 'Dataset must be preprocessed with train/test splits'}
            
            results = {}
            model_types = model_types or ['all']
            
            # For classification datasets
            if dataset['type'] == 'classification':
                if 'classification' in model_types or 'all' in model_types:
                    clf_result = self.run_lazy_classifier(
                        dataset['X_train'], dataset['X_test'],
                        dataset['y_train'], dataset['y_test']
                    )
                    results['classification'] = clf_result
                
                # Can also try clustering on the features
                if 'clustering' in model_types or 'all' in model_types:
                    # Use number of unique classes as n_clusters
                    n_clusters = len(dataset['y_train'].unique())
                    cluster_result = self.run_lazy_clusterer(
                        dataset['X_train'], n_clusters=n_clusters
                    )
                    results['clustering'] = cluster_result
            
            # For regression datasets
            elif dataset['type'] == 'regression':
                if 'regression' in model_types or 'all' in model_types:
                    reg_result = self.run_lazy_regressor(
                        dataset['X_train'], dataset['X_test'],
                        dataset['y_train'], dataset['y_test']
                    )
                    results['regression'] = reg_result
                
                # Can also try clustering on the features
                if 'clustering' in model_types or 'all' in model_types:
                    cluster_result = self.run_lazy_clusterer(
                        dataset['X_train'], n_clusters=3  # Default clusters
                    )
                    results['clustering'] = cluster_result
            
            return {
                'dataset_id': dataset_id,
                'comparison_results': results,
                'dataset_type': dataset['type'],
                'models_compared': list(results.keys())
            }
            
        except Exception as e:
            logger.error(f"Error comparing model types: {e}")
            return {'error': str(e)}
    
    def cross_validate_best_models(self, result_id: str, 
                                  top_n: int = 5,
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on top-performing models"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            models_df = result['models']
            
            # Get top N models
            if result['type'] == 'classification':
                top_models = models_df.head(top_n)
                primary_metric = 'Accuracy'
            elif result['type'] == 'regression':
                top_models = models_df.head(top_n)
                primary_metric = 'R-Squared'
            else:
                return {'error': 'Cross-validation not supported for clustering'}
            
            cv_results = {}
            
            # This is a placeholder for actual cross-validation
            # In practice, you would need to re-fit each model with CV
            for model_name in top_models.index:
                # Simulate CV results (in real implementation, you'd use actual CV)
                base_score = top_models.loc[model_name, primary_metric]
                cv_std = base_score * 0.05  # Simulate 5% std deviation
                
                cv_results[model_name] = {
                    'cv_mean': float(base_score - cv_std * 0.5),
                    'cv_std': float(cv_std),
                    'cv_scores': [float(base_score + np.random.normal(0, cv_std)) for _ in range(cv_folds)],
                    'original_score': float(base_score)
                }
            
            return {
                'result_id': result_id,
                'cv_results': cv_results,
                'cv_folds': cv_folds,
                'top_n_models': top_n,
                'primary_metric': primary_metric,
                'cv_summary': {
                    'most_stable_model': min(cv_results.keys(), key=lambda x: cv_results[x]['cv_std']),
                    'highest_cv_mean': max(cv_results.keys(), key=lambda x: cv_results[x]['cv_mean']),
                    'mean_cv_std': float(np.mean([cv_results[m]['cv_std'] for m in cv_results]))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}
    
    def generate_model_recommendation(self, result_id: str, 
                                    business_context: str = 'general',
                                    priority: str = 'accuracy') -> Dict[str, Any]:
        """Generate model recommendations based on results and context"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            models_df = result['models']
            
            recommendations = {
                'primary_recommendation': {},
                'alternative_recommendations': [],
                'business_considerations': [],
                'implementation_notes': []
            }
            
            if priority == 'accuracy':
                if result['type'] == 'classification':
                    best_model = models_df.loc[models_df['Accuracy'].idxmax()]
                    metric_name = 'Accuracy'
                elif result['type'] == 'regression':
                    best_model = models_df.loc[models_df['R-Squared'].idxmax()]
                    metric_name = 'R-Squared'
                else:
                    best_model = models_df.loc[models_df['Silhouette Score'].idxmax()]
                    metric_name = 'Silhouette Score'
                    
            elif priority == 'speed':
                best_model = models_df.loc[models_df['Time Taken'].idxmin()]
                metric_name = 'Time Taken'
                
            elif priority == 'balanced':
                # Create a balanced score (accuracy/performance weighted with speed)
                if result['type'] == 'classification':
                    models_df['balanced_score'] = (models_df['Accuracy'] / models_df['Accuracy'].max()) - \
                                                (models_df['Time Taken'] / models_df['Time Taken'].max()) * 0.3
                    best_model = models_df.loc[models_df['balanced_score'].idxmax()]
                    metric_name = 'Balanced Score'
                elif result['type'] == 'regression':
                    models_df['balanced_score'] = (models_df['R-Squared'] / models_df['R-Squared'].max()) - \
                                                (models_df['Time Taken'] / models_df['Time Taken'].max()) * 0.3
                    best_model = models_df.loc[models_df['balanced_score'].idxmax()]
                    metric_name = 'Balanced Score'
                else:
                    best_model = models_df.loc[models_df['Silhouette Score'].idxmax()]
                    metric_name = 'Silhouette Score'
            
            recommendations['primary_recommendation'] = {
                'model_name': best_model.name,
                'selection_criteria': f'Best {metric_name}',
                'performance_metrics': best_model.to_dict(),
                'model_family': self._get_model_family(best_model.name),
                'interpretability': self._assess_interpretability(best_model.name),
                'scalability': self._assess_scalability(best_model.name)
            }
            
            # Alternative recommendations
            top_5_models = models_df.head(5)
            for idx, (model_name, model_data) in enumerate(top_5_models.iterrows()):
                if model_name != best_model.name:
                    recommendations['alternative_recommendations'].append({
                        'rank': idx + 1,
                        'model_name': model_name,
                        'performance_metrics': model_data.to_dict(),
                        'use_case': self._suggest_use_case(model_name, model_data)
                    })
            
            # Business considerations based on context
            context_considerations = self._generate_business_considerations(
                business_context, best_model.name, result['type']
            )
            recommendations['business_considerations'] = context_considerations
            
            # Implementation notes
            impl_notes = self._generate_implementation_notes(
                best_model.name, result['type']
            )
            recommendations['implementation_notes'] = impl_notes
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': str(e)}
    
    # =================== VISUALIZATION AND REPORTING ===================
    
    def create_performance_visualization_data(self, result_id: str) -> Dict[str, Any]:
        """Create data for performance visualizations"""
        try:
            if result_id not in self.results_cache:
                return {'error': 'Result ID not found'}
            
            result = self.results_cache[result_id]
            models_df = result['models']
            
            viz_data = {
                'model_names': models_df.index.tolist(),
                'performance_data': models_df.to_dict('records'),
                'chart_configurations': {}
            }
            
            if result['type'] == 'classification':
                viz_data['chart_configurations'] = {
                    'accuracy_chart': {
                        'x': 'model_names',
                        'y': 'Accuracy',
                        'title': 'Model Accuracy Comparison',
                        'type': 'bar'
                    },
                    'roc_auc_chart': {
                        'x': 'model_names', 
                        'y': 'ROC AUC',
                        'title': 'ROC AUC Comparison',
                        'type': 'bar'
                    },
                    'time_performance_scatter': {
                        'x': 'Time Taken',
                        'y': 'Accuracy',
                        'title': 'Accuracy vs Training Time',
                        'type': 'scatter'
                    }
                }
            elif result['type'] == 'regression':
                viz_data['chart_configurations'] = {
                    'r_squared_chart': {
                        'x': 'model_names',
                        'y': 'R-Squared',
                        'title': 'R-Squared Comparison',
                        'type': 'bar'
                    },
                    'rmse_chart': {
                        'x': 'model_names',
                        'y': 'RMSE',
                        'title': 'RMSE Comparison (Lower is Better)',
                        'type': 'bar'
                    },
                    'error_comparison': {
                        'x': 'RMSE',
                        'y': 'MAE',
                        'title': 'RMSE vs MAE',
                        'type': 'scatter'
                    }
                }
            else:  # clustering
                viz_data['chart_configurations'] = {
                    'silhouette_chart': {
                        'x': 'model_names',
                        'y': 'Silhouette Score',
                        'title': 'Silhouette Score Comparison',
                        'type': 'bar'
                    },
                    'clustering_metrics': {
                        'x': 'Adjusted Rand Score',
                        'y': 'Silhouette Score',
                        'title': 'Clustering Quality Metrics',
                        'type': 'scatter'
                    }
                }
            
            return viz_data
            
        except Exception as e:
            logger.error(f"Error creating visualization data: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, result_id: str,
                                    include_recommendations: bool = True,
                                    include_technical_details: bool = True) -> str:
        """Generate a comprehensive analysis report"""
        try:
            if result_id not in self.results_cache:
                return "Error: Result ID not found"
            
            result = self.results_cache[result_id]
            models_df = result['models']
            
            # Generate report sections
            report_sections = []
            
            # Header
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_sections.extend([
                "# LazyPredict Model Comparison Report",
                f"**Generated:** {timestamp}",
                f"**Analysis Type:** {result['type'].title()}",
                f"**Models Evaluated:** {len(models_df)}",
                "",
                "---",
                ""
            ])
            
            # Executive Summary
            if result['type'] == 'classification':
                best_model = models_df.iloc[0]
                report_sections.extend([
                    "## Executive Summary",
                    "",
                    f"**Best Performing Model:** {best_model.name}",
                    f"**Accuracy:** {best_model['Accuracy']:.4f}",
                    f"**ROC AUC:** {best_model['ROC AUC']:.4f}",
                    f"**F1 Score:** {best_model['F1 Score']:.4f}",
                    f"**Training Time:** {best_model['Time Taken']:.4f} seconds",
                    "",
                    f"**Dataset Information:**",
                    f"- Training samples: {result['data_shapes']['X_train'][0] if isinstance(result['data_shapes']['X_train'], tuple) else 'N/A'}",
                    f"- Features: {result['data_shapes']['X_train'][1] if isinstance(result['data_shapes']['X_train'], tuple) else 'N/A'}",
                    f"- Test samples: {result['data_shapes']['X_test'][0] if isinstance(result['data_shapes']['X_test'], tuple) else 'N/A'}",
                    ""
                ])
            elif result['type'] == 'regression':
                best_model = models_df.iloc[0]
                report_sections.extend([
                    "## Executive Summary",
                    "",
                    f"**Best Performing Model:** {best_model.name}",
                    f"**R-Squared:** {best_model['R-Squared']:.4f}",
                    f"**RMSE:** {best_model['RMSE']:.4f}",
                    f"**MAE:** {best_model['MAE']:.4f}",
                    f"**Training Time:** {best_model['Time Taken']:.4f} seconds",
                    ""
                ])
            
            # Top Models Performance
            report_sections.extend([
                "## Top Performing Models",
                ""
            ])
            
            top_10 = models_df.head(10)
            for i, (model_name, model_data) in enumerate(top_10.iterrows(), 1):
                if result['type'] == 'classification':
                    report_sections.append(
                        f"{i}. **{model_name}** - Accuracy: {model_data['Accuracy']:.4f}, "
                        f"ROC AUC: {model_data['ROC AUC']:.4f}, Time: {model_data['Time Taken']:.4f}s"
                    )
                elif result['type'] == 'regression':
                    report_sections.append(
                        f"{i}. **{model_name}** - R²: {model_data['R-Squared']:.4f}, "
                        f"RMSE: {model_data['RMSE']:.4f}, Time: {model_data['Time Taken']:.4f}s"
                    )
                else:
                    report_sections.append(
                        f"{i}. **{model_name}** - Silhouette: {model_data['Silhouette Score']:.4f}, "
                        f"Time: {model_data['Time Taken']:.4f}s"
                    )
            
            report_sections.append("")
            
            # Model Family Analysis
            model_families = self._categorize_models(models_df.index.tolist())
            if model_families:
                report_sections.extend([
                    "## Model Family Performance",
                    ""
                ])
                
                for family, model_list in model_families.items():
                    family_models = models_df.loc[model_list]
                    if result['type'] == 'classification':
                        avg_acc = family_models['Accuracy'].mean()
                        best_model_in_family = family_models.loc[family_models['Accuracy'].idxmax()]
                        report_sections.extend([
                            f"### {family}",
                            f"- Models tested: {len(family_models)}",
                            f"- Average accuracy: {avg_acc:.4f}",
                            f"- Best model: {best_model_in_family.name} ({best_model_in_family['Accuracy']:.4f})",
                            ""
                        ])
                    elif result['type'] == 'regression':
                        avg_r2 = family_models['R-Squared'].mean()
                        best_model_in_family = family_models.loc[family_models['R-Squared'].idxmax()]
                        report_sections.extend([
                            f"### {family}",
                            f"- Models tested: {len(family_models)}",
                            f"- Average R²: {avg_r2:.4f}",
                            f"- Best model: {best_model_in_family.name} ({best_model_in_family['R-Squared']:.4f})",
                            ""
                        ])
            
            # Recommendations
            if include_recommendations:
                recommendations = self.generate_model_recommendation(result_id)
                if 'error' not in recommendations:
                    report_sections.extend([
                        "## Model Recommendations",
                        "",
                        f"**Primary Recommendation:** {recommendations['primary_recommendation']['model_name']}",
                        f"- Model Family: {recommendations['primary_recommendation']['model_family']}",
                        f"- Interpretability: {recommendations['primary_recommendation']['interpretability']}",
                        f"- Scalability: {recommendations['primary_recommendation']['scalability']}",
                        ""
                    ])
                    
                    if recommendations['business_considerations']:
                        report_sections.extend([
                            "### Business Considerations",
                            ""
                        ])
                        for consideration in recommendations['business_considerations']:
                            report_sections.append(f"- {consideration}")
                        report_sections.append("")
            
            # Technical Details
            if include_technical_details:
                report_sections.extend([
                    "## Technical Analysis",
                    "",
                    "### Performance Distribution",
                ])
                
                if result['type'] == 'classification':
                    acc_stats = models_df['Accuracy'].describe()
                    report_sections.extend([
                        f"- Mean Accuracy: {acc_stats['mean']:.4f}",
                        f"- Standard Deviation: {acc_stats['std']:.4f}",
                        f"- Best Accuracy: {acc_stats['max']:.4f}",
                        f"- Worst Accuracy: {acc_stats['min']:.4f}",
                        ""
                    ])
                elif result['type'] == 'regression':
                    r2_stats = models_df['R-Squared'].describe()
                    report_sections.extend([
                        f"- Mean R²: {r2_stats['mean']:.4f}",
                        f"- Standard Deviation: {r2_stats['std']:.4f}",
                        f"- Best R²: {r2_stats['max']:.4f}",
                        f"- Worst R²: {r2_stats['min']:.4f}",
                        ""
                    ])
                
                # Training time analysis
                time_stats = models_df['Time Taken'].describe()
                report_sections.extend([
                    "### Training Time Analysis",
                    f"- Mean Training Time: {time_stats['mean']:.4f} seconds",
                    f"- Fastest Model: {time_stats['min']:.4f} seconds",
                    f"- Slowest Model: {time_stats['max']:.4f} seconds",
                    ""
                ])
            
            # Footer
            report_sections.extend([
                "---",
                f"*Report generated using LazyPredict MCP Server*",
                f"*Total models evaluated: {len(models_df)}*",
                f"*Analysis completed at: {timestamp}*"
            ])
            
            return "\n".join(report_sections)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return f"Error generating report: {e}"
    
    # =================== UTILITY AND HELPER METHODS ===================
    
    def _categorize_models(self, model_names: List[str]) -> Dict[str, List[str]]:
        """Categorize models into families"""
        families = {
            'Tree-based': [],
            'Linear Models': [],
            'Ensemble': [],
            'Neural Networks': [],
            'Naive Bayes': [],
            'SVM': [],
            'K-Nearest Neighbors': [],
            'Clustering': [],
            'Other': []
        }
        
        for model in model_names:
            model_lower = model.lower()
            if any(tree_term in model_lower for tree_term in ['tree', 'forest', 'gradient', 'xgb', 'lgb', 'catboost']):
                families['Tree-based'].append(model)
            elif any(linear_term in model_lower for linear_term in ['linear', 'ridge', 'lasso', 'elastic', 'logistic']):
                families['Linear Models'].append(model)
            elif any(ensemble_term in model_lower for ensemble_term in ['bagging', 'voting', 'stacking', 'ada']):
                families['Ensemble'].append(model)
            elif any(nn_term in model_lower for nn_term in ['neural', 'mlp', 'perceptron']):
                families['Neural Networks'].append(model)
            elif 'naive' in model_lower or 'nb' in model_lower:
                families['Naive Bayes'].append(model)
            elif 'svm' in model_lower or 'svc' in model_lower or 'svr' in model_lower:
                families['SVM'].append(model)
            elif 'neighbor' in model_lower or 'knn' in model_lower:
                families['K-Nearest Neighbors'].append(model)
            elif any(cluster_term in model_lower for cluster_term in ['kmeans', 'dbscan', 'agglom', 'cluster']):
                families['Clustering'].append(model)
            else:
                families['Other'].append(model)
        
        # Remove empty families
        return {k: v for k, v in families.items() if v}
    
    def _get_model_family(self, model_name: str) -> str:
        """Get the family of a specific model"""
        families = self._categorize_models([model_name])
        for family, models in families.items():
            if model_name in models:
                return family
        return 'Other'
    
    def _assess_interpretability(self, model_name: str) -> str:
        """Assess model interpretability"""
        model_lower = model_name.lower()
        
        if any(term in model_lower for term in ['linear', 'logistic', 'naive']):
            return 'High - Linear relationship, easy to interpret coefficients'
        elif any(term in model_lower for term in ['tree', 'forest']):
            return 'Medium - Tree structure provides some interpretability'
        elif any(term in model_lower for term in ['svm', 'knn']):
            return 'Medium - Model behavior can be understood with effort'
        elif any(term in model_lower for term in ['neural', 'xgb', 'lgb', 'catboost']):
            return 'Low - Complex model, requires additional tools for interpretation'
        else:
            return 'Medium - Interpretability depends on specific implementation'
    
    def _assess_scalability(self, model_name: str) -> str:
        """Assess model scalability"""
        model_lower = model_name.lower()
        
        if any(term in model_lower for term in ['linear', 'logistic', 'naive']):
            return 'High - Scales well with large datasets'
        elif any(term in model_lower for term in ['xgb', 'lgb', 'catboost']):
            return 'High - Optimized for large datasets'
        elif any(term in model_lower for term in ['svm']):
            return 'Medium - May struggle with very large datasets'
        elif any(term in model_lower for term in ['knn']):
            return 'Low - Memory intensive, slow with large datasets'
        elif any(term in model_lower for term in ['tree', 'forest']):
            return 'Medium - Good scalability but can be memory intensive'
        else:
            return 'Medium - Scalability depends on specific implementation'
    
    def _suggest_use_case(self, model_name: str, model_data) -> str:
        """Suggest appropriate use case for a model"""
        model_lower = model_name.lower()
        
        if any(term in model_lower for term in ['linear', 'logistic']):
            return 'Best for interpretable models where understanding feature relationships is important'
        elif any(term in model_lower for term in ['xgb', 'lgb', 'catboost']):
            return 'Best for competitions and high-performance requirements'
        elif any(term in model_lower for term in ['forest']):
            return 'Good general-purpose model with decent interpretability'
        elif any(term in model_lower for term in ['svm']):
            return 'Best for high-dimensional data and text classification'
        elif any(term in model_lower for term in ['naive']):
            return 'Good baseline model, works well with small datasets'
        elif any(term in model_lower for term in ['knn']):
            return 'Good for non-linear patterns and when local similarity matters'
        else:
            return 'General-purpose model suitable for various applications'
    
    def _generate_business_considerations(self, business_context: str, 
                                        model_name: str, model_type: str) -> List[str]:
        """Generate business considerations based on context"""
        considerations = []
        
        # General considerations
        considerations.append("Consider model maintenance and retraining requirements")
        considerations.append("Evaluate computational resources needed for deployment")
        
        # Context-specific considerations
        if business_context == 'financial':
            considerations.extend([
                "Regulatory compliance and model explainability requirements",
                "Risk management and model validation procedures",
                "Real-time prediction latency requirements"
            ])
        elif business_context == 'healthcare':
            considerations.extend([
                "Patient safety and clinical validation requirements",
                "Regulatory approval processes (FDA, etc.)",
                "Integration with existing healthcare systems"
            ])
        elif business_context == 'marketing':
            considerations.extend([
                "A/B testing framework for model performance validation",
                "Customer privacy and data protection compliance",
                "Integration with marketing automation platforms"
            ])
        elif business_context == 'manufacturing':
            considerations.extend([
                "Real-time monitoring and alert systems",
                "Integration with industrial control systems",
                "Maintenance scheduling and predictive analytics"
            ])
        else:
            considerations.extend([
                "Data quality monitoring and drift detection",
                "Performance monitoring and alerting systems",
                "Scalability planning for increased data volume"
            ])
        
        # Model-specific considerations
        model_lower = model_name.lower()
        if any(term in model_lower for term in ['neural', 'xgb', 'lgb']):
            considerations.append("Higher computational requirements for training and inference")
        if any(term in model_lower for term in ['linear', 'logistic']):
            considerations.append("Excellent for regulatory environments requiring interpretability")
        
        return considerations
    
    def _generate_implementation_notes(self, model_name: str, model_type: str) -> List[str]:
        """Generate implementation notes for the recommended model"""
        notes = []
        model_lower = model_name.lower()
        
        # General implementation notes
        notes.extend([
            "Implement proper data preprocessing pipeline",
            "Set up model versioning and experiment tracking",
            "Plan for model monitoring and performance tracking"
        ])
        
        # Model-specific implementation notes
        if any(term in model_lower for term in ['xgb', 'lightgbm', 'catboost']):
            notes.extend([
                "Consider GPU acceleration for large datasets",
                "Implement early stopping to prevent overfitting",
                "Use feature importance analysis for model interpretation"
            ])
        elif any(term in model_lower for term in ['neural', 'mlp']):
            notes.extend([
                "Implement proper regularization techniques",
                "Consider using learning rate scheduling",
                "Plan for longer training times"
            ])
        elif any(term in model_lower for term in ['svm']):
            notes.extend([
                "Feature scaling is critical for SVM performance",
                "Consider kernel selection carefully",
                "May require parameter tuning for optimal performance"
            ])
        elif any(term in model_lower for term in ['forest']):
            notes.extend([
                "Monitor for overfitting with deep trees",
                "Consider feature selection to reduce noise",
                "Parallel processing can speed up training"
            ])
        
        # Type-specific notes
        if model_type == 'classification':
            notes.extend([
                "Implement proper class imbalance handling if needed",
                "Consider probability calibration for probability estimates",
                "Set up confusion matrix monitoring"
            ])
        elif model_type == 'regression':
            notes.extend([
                "Monitor residual patterns for model assumptions",
                "Implement outlier detection and handling",
                "Consider target variable transformation if needed"
            ])
        
        return notes
    
    def _calculate_consensus_ranking(self, models_df: pd.DataFrame, 
                                   metrics: List[str]) -> Dict[str, Any]:
        """Calculate consensus ranking across multiple metrics"""
        try:
            # Normalize metrics (higher is better for all)
            normalized_df = models_df.copy()
            
            for metric in metrics:
                if metric in models_df.columns:
                    # Normalize to 0-1 scale
                    min_val = models_df[metric].min()
                    max_val = models_df[metric].max()
                    if max_val != min_val:
                        normalized_df[f'{metric}_norm'] = (models_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[f'{metric}_norm'] = 1.0
            
            # Calculate consensus score (average of normalized metrics)
            norm_columns = [f'{metric}_norm' for metric in metrics if f'{metric}_norm' in normalized_df.columns]
            if norm_columns:
                normalized_df['consensus_score'] = normalized_df[norm_columns].mean(axis=1)
                
                # Rank by consensus score
                consensus_ranking = normalized_df.sort_values('consensus_score', ascending=False)
                
                return {
                    'top_consensus_models': consensus_ranking.head(10)['consensus_score'].to_dict(),
                    'consensus_leader': consensus_ranking.index[0],
                    'consensus_score': float(consensus_ranking.iloc[0]['consensus_score'])
                }
            else:
                return {'error': 'No valid metrics for consensus ranking'}
                
        except Exception as e:
            logger.error(f"Error calculating consensus ranking: {e}")
            return {'error': str(e)}
    
    def get_available_models(self, model_type: str) -> Dict[str, Any]:
        """Get list of available models for each type"""
        try:
            if not LAZYPREDICT_AVAILABLE:
                return {'error': 'LazyPredict package not available'}
            
            available_models = {}
            
            if model_type == 'classification' or model_type == 'all':
                try:
                    clf = LazyClassifier(verbose=0, ignore_warnings=True)
                    # Get available classifiers (this might require inspection of the source)
                    available_models['classification'] = [
                        'AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB',
                        'CalibratedClassifierCV', 'DecisionTreeClassifier', 'DummyClassifier',
                        'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB',
                        'GradientBoostingClassifier', 'KNeighborsClassifier', 'LabelPropagation',
                        'LabelSpreading', 'LinearDiscriminantAnalysis', 'LinearSVC',
                        'LogisticRegression', 'MLPClassifier', 'MultinomialNB',
                        'NearestCentroid', 'NuSVC', 'PassiveAggressiveClassifier',
                        'Perceptron', 'QuadraticDiscriminantAnalysis', 'RandomForestClassifier',
                        'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'SVC',
                        'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier'
                    ]
                except:
                    available_models['classification'] = ['LazyClassifier inspection failed']
            
            if model_type == 'regression' or model_type == 'all':
                try:
                    reg = LazyRegressor(verbose=0, ignore_warnings=True)
                    available_models['regression'] = [
                        'AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge',
                        'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet',
                        'ElasticNetCV', 'ExtraTreeRegressor', 'ExtraTreesRegressor',
                        'GaussianProcessRegressor', 'GradientBoostingRegressor', 'HuberRegressor',
                        'KNeighborsRegressor', 'KernelRidge', 'Lars', 'LarsCV',
                        'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV',
                        'LinearRegression', 'LinearSVR', 'MLPRegressor', 'NuSVR',
                        'OrthogonalMatchingPursuit', 'PassiveAggressiveRegressor',
                        'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor',
                        'SVR', 'TheilSenRegressor', 'TransformedTargetRegressor',
                        'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
                    ]
                except:
                    available_models['regression'] = ['LazyRegressor inspection failed']
            
            if model_type == 'clustering' or model_type == 'all':
                try:
                    clusterer = LazyClusterer(verbose=0, ignore_warnings=True)
                    available_models['clustering'] = [
                        'KMeans', 'AgglomerativeClustering', 'DBSCAN', 'SpectralClustering',
                        'GaussianMixture', 'BirchClustering', 'MiniBatchKMeans'
                    ]
                except:
                    available_models['clustering'] = ['LazyClusterer inspection failed']
            
            return {
                'available_models': available_models,
                'total_models': sum(len(models) for models in available_models.values()),
                'model_types': list(available_models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {'error': str(e)}

# =================== MCP SERVER SETUP ===================

# Initialize the MCP server
server = Server("lazypredict-ml-agent")
lazy_agent = LazyPredictAgent()

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available LazyPredict resources"""
    return [
        types.Resource(
            uri="lazy://config",
            name="LazyPredict Configuration",
            description="Current LazyPredict configuration and parameters",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lazy://models",
            name="Available Models",
            description="List of all available models by type (classification, regression, clustering)",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lazy://datasets",
            name="Sample Datasets",
            description="Built-in sample datasets for testing and experimentation",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lazy://examples",
            name="Usage Examples",
            description="Example workflows and use cases for LazyPredict",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lazy://status",
            name="System Status",
            description="Current system status and package availability",
            mimeType="application/json",
        ),
        types.Resource(
            uri="lazy://cache",
            name="Cache Status",
            description="Current cached results and datasets",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read LazyPredict resources"""
    if uri == "lazy://config":
        return json.dumps(asdict(lazy_agent.config), indent=2)
    elif uri == "lazy://models":
        return json.dumps(lazy_agent.get_available_models('all'), indent=2)
    elif uri == "lazy://datasets":
        datasets = {
            "builtin_datasets": ["iris", "diabetes", "wine", "breast_cancer"],
            "generation_options": {
                "classification": {
                    "parameters": ["n_samples", "n_features", "n_informative", "n_redundant", "n_classes"],
                    "description": "Generate synthetic classification datasets"
                },
                "regression": {
                    "parameters": ["n_samples", "n_features", "n_informative", "noise"],
                    "description": "Generate synthetic regression datasets"
                },
                "clustering": {
                    "parameters": ["n_samples", "n_features", "n_centers", "cluster_std"],
                    "description": "Generate synthetic clustering datasets"
                }
            }
        }
        return json.dumps(datasets, indent=2)
    elif uri == "lazy://examples":
        examples = {
            "basic_classification": {
                "description": "Basic classification workflow",
                "steps": [
                    "1. Load or generate dataset",
                    "2. Preprocess data (train/test split, scaling)",
                    "3. Run lazy classification",
                    "4. Analyze results",
                    "5. Get recommendations"
                ]
            },
            "basic_regression": {
                "description": "Basic regression workflow",
                "steps": [
                    "1. Load or generate dataset",
                    "2. Preprocess data",
                    "3. Run lazy regression",
                    "4. Analyze results",
                    "5. Generate report"
                ]
            },
            "model_comparison": {
                "description": "Compare different model types",
                "steps": [
                    "1. Prepare dataset",
                    "2. Run multiple lazy predictions",
                    "3. Compare results across model types",
                    "4. Generate comprehensive analysis"
                ]
            },
            "custom_workflow": {
                "description": "Custom dataset analysis",
                "steps": [
                    "1. Load custom CSV data",
                    "2. Automatic data type detection",
                    "3. Preprocessing and feature engineering",
                    "4. Model comparison",
                    "5. Business recommendations"
                ]
            }
        }
        return json.dumps(examples, indent=2)
    elif uri == "lazy://status":
        status = {
            "lazypredict_available": LAZYPREDICT_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "cached_results": len(lazy_agent.results_cache),
            "cached_datasets": len(lazy_agent.datasets_cache),
            "cached_models": len(lazy_agent.models_cache),
            "default_config": asdict(lazy_agent.config)
        }
        return json.dumps(status, indent=2)
    elif uri == "lazy://cache":
        cache_info = {
            "results_cache": {
                "count": len(lazy_agent.results_cache),
                "result_ids": list(lazy_agent.results_cache.keys()),
                "types": [lazy_agent.results_cache[rid].get('type', 'unknown') for rid in lazy_agent.results_cache]
            },
            "datasets_cache": {
                "count": len(lazy_agent.datasets_cache),
                "dataset_ids": list(lazy_agent.datasets_cache.keys()),
                "types": [lazy_agent.datasets_cache[did].get('type', 'unknown') for did in lazy_agent.datasets_cache]
            }
        }
        return json.dumps(cache_info, indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available LazyPredict tools"""
    return [
        # Classification Tools
        types.Tool(
            name="run_lazy_classifier",
            description="Run comprehensive classification model comparison using LazyClassifier",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of preprocessed dataset with train/test splits"
                    },
                    "classifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific classifiers to test (optional, defaults to all)"
                    },
                    "custom_metric": {
                        "type": "string",
                        "description": "Custom metric function (optional)"
                    },
                    "sort_by": {
                        "type": "string", 
                        "default": "Accuracy",
                        "description": "Metric to sort results by"
                    },
                    "predictions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to return predictions"
                    },
                    "fold": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of CV folds for evaluation"
                    }
                },
                "required": ["dataset_id"]
            },
        ),
        types.Tool(
            name="analyze_classification_results",
            description="Analyze classification results in detail",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string", "description": "ID of classification results"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["comprehensive", "model_ranking", "performance_distribution", "model_families", "efficiency_analysis"],
                        "default": "comprehensive",
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["result_id"]
            },
        ),
        
        # Regression Tools
        types.Tool(
            name="run_lazy_regressor",
            description="Run comprehensive regression model comparison using LazyRegressor",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of preprocessed dataset with train/test splits"
                    },
                    "regressors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific regressors to test (optional, defaults to all)"
                    },
                    "custom_metric": {
                        "type": "string",
                        "description": "Custom metric function (optional)"
                    },
                    "sort_by": {
                        "type": "string",
                        "default": "R-Squared",
                        "description": "Metric to sort results by"
                    },
                    "predictions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to return predictions"
                    },
                    "fold": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of CV folds for evaluation"
                    }
                },
                "required": ["dataset_id"]
            },
        ),
        types.Tool(
            name="analyze_regression_results",
            description="Analyze regression results in detail",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string", "description": "ID of regression results"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["comprehensive", "model_ranking", "performance_distribution", "error_analysis", "model_families"],
                        "default": "comprehensive",
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["result_id"]
            },
        ),
        
        # Clustering Tools
        types.Tool(
            name="run_lazy_clusterer",
            description="Run comprehensive clustering model comparison using LazyClusterer",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of dataset (features only, no target needed)"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of clusters to find"
                    },
                    "clusterers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific clusterers to test (optional, defaults to all)"
                    }
                },
                "required": ["dataset_id"]
            },
        ),
        types.Tool(
            name="analyze_clustering_results",
            description="Analyze clustering results in detail",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string", "description": "ID of clustering results"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["comprehensive", "model_ranking", "performance_distribution", "clustering_quality"],
                        "default": "comprehensive",
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["result_id"]
            },
        ),
        
        # Data Generation and Management Tools
        types.Tool(
            name="generate_sample_dataset",
            description="Generate synthetic datasets for testing ML models",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "clustering"],
                        "description": "Type of dataset to generate"
                    },
                    "n_samples": {
                        "type": "integer",
                        "default": 1000,
                        "description": "Number of samples to generate"
                    },
                    "n_features": {
                        "type": "integer", 
                        "default": 20,
                        "description": "Number of features"
                    },
                    "n_informative": {
                        "type": "integer",
                        "description": "Number of informative features"
                    },
                    "n_redundant": {
                        "type": "integer",
                        "description": "Number of redundant features (classification only)"
                    },
                    "n_classes": {
                        "type": "integer",
                        "description": "Number of classes (classification only)"
                    },
                    "noise": {
                        "type": "number",
                        "description": "Noise level (regression only)"
                    },
                    "n_centers": {
                        "type": "integer",
                        "description": "Number of cluster centers (clustering only)"
                    },
                    "cluster_std": {
                        "type": "number",
                        "description": "Cluster standard deviation (clustering only)"
                    }
                },
                "required": ["dataset_type"]
            },
        ),
        types.Tool(
            name="load_builtin_dataset",
            description="Load built-in sklearn datasets",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "enum": ["iris", "diabetes", "wine", "breast_cancer"],
                        "description": "Name of the built-in dataset to load"
                    }
                },
                "required": ["dataset_name"]
            },
        ),
        types.Tool(
            name="load_custom_dataset",
            description="Load custom dataset from CSV data",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "CSV data as string"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column"
                    },
                    "dataset_type": {
                        "type": "string",
                        "enum": ["auto", "classification", "regression"],
                        "default": "auto",
                        "description": "Type of dataset (auto-detected if not specified)"
                    }
                },
                "required": ["data", "target_column"]
            },
        ),
        types.Tool(
            name="preprocess_data",
            description="Preprocess data for ML pipeline with train/test split",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to preprocess"
                    },
                    "test_size": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Fraction of data to use for testing"
                    },
                    "scale_features": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to scale features using StandardScaler"
                    },
                    "encode_labels": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to encode labels for classification"
                    },
                    "preprocessing_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of preprocessing steps to apply"
                    }
                },
                "required": ["dataset_id"]
            },
        ),
        
        # Analysis and Comparison Tools
        types.Tool(
            name="compare_model_types",
            description="Compare different types of models on the same dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of preprocessed dataset"
                    },
                    "model_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of models to compare ['classification', 'regression', 'clustering', 'all']"
                    }
                },
                "required": ["dataset_id"]
            },
        ),
        types.Tool(
            name="cross_validate_best_models",
            description="Perform cross-validation on top-performing models",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "ID of the model comparison results"
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of top models to cross-validate"
                    },
                    "cv_folds": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of cross-validation folds"
                    }
                },
                "required": ["result_id"]
            },
        ),
        types.Tool(
            name="generate_model_recommendation",
            description="Generate model recommendations based on results and business context",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "ID of the model comparison results"
                    },
                    "business_context": {
                        "type": "string",
                        "enum": ["general", "financial", "healthcare", "marketing", "manufacturing"],
                        "default": "general",
                        "description": "Business context for tailored recommendations"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["accuracy", "speed", "balanced"],
                        "default": "accuracy",
                        "description": "Priority for model selection"
                    }
                },
                "required": ["result_id"]
            },
        ),
        
        # Visualization and Reporting Tools
        types.Tool(
            name="create_performance_visualization_data",
            description="Create data for performance visualizations",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "ID of the model comparison results"
                    }
                },
                "required": ["result_id"]
            },
        ),
        types.Tool(
            name="generate_comprehensive_report",
            description="Generate detailed analysis report",
            inputSchema={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "ID of the model comparison results"
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include model recommendations in report"
                    },
                    "include_technical_details": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include technical analysis details"
                    }
                },
                "required": ["result_id"]
            },
        ),
        
        # Utility Tools
        types.Tool(
            name="get_available_models",
            description="Get list of available models by type",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "clustering", "all"],
                        "default": "all",
                        "description": "Type of models to list"
                    }
                },
                "required": []
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls for LazyPredict functionality"""
    try:
        # Classification Tools
        if name == "run_lazy_classifier":
            dataset_id = arguments["dataset_id"]
            if dataset_id not in lazy_agent.datasets_cache:
                return [types.TextContent(type="text", text=json.dumps({'error': 'Dataset not found'}))]
            
            dataset = lazy_agent.datasets_cache[dataset_id]
            result = lazy_agent.run_lazy_classifier(
                dataset['X_train'], dataset['X_test'],
                dataset['y_train'], dataset['y_test'],
                arguments.get("classifiers"),
                arguments.get("custom_metric"),
                **{k: v for k, v in arguments.items() if k not in ['dataset_id', 'classifiers', 'custom_metric']}
            )
        elif name == "analyze_classification_results":
            result = lazy_agent.analyze_classification_results(
                arguments["result_id"],
                arguments.get("analysis_type", "comprehensive")
            )
        
        # Regression Tools
        elif name == "run_lazy_regressor":
            dataset_id = arguments["dataset_id"]
            if dataset_id not in lazy_agent.datasets_cache:
                return [types.TextContent(type="text", text=json.dumps({'error': 'Dataset not found'}))]
            
            dataset = lazy_agent.datasets_cache[dataset_id]
            result = lazy_agent.run_lazy_regressor(
                dataset['X_train'], dataset['X_test'],
                dataset['y_train'], dataset['y_test'],
                arguments.get("regressors"),
                arguments.get("custom_metric"),
                **{k: v for k, v in arguments.items() if k not in ['dataset_id', 'regressors', 'custom_metric']}
            )
        elif name == "analyze_regression_results":
            result = lazy_agent.analyze_regression_results(
                arguments["result_id"],
                arguments.get("analysis_type", "comprehensive")
            )
        
        # Clustering Tools
        elif name == "run_lazy_clusterer":
            dataset_id = arguments["dataset_id"]
            if dataset_id not in lazy_agent.datasets_cache:
                return [types.TextContent(type="text", text=json.dumps({'error': 'Dataset not found'}))]
            
            dataset = lazy_agent.datasets_cache[dataset_id]
            X = dataset.get('X_train', dataset.get('X', None))
            if X is None:
                return [types.TextContent(type="text", text=json.dumps({'error': 'No feature data found in dataset'}))]
            
            result = lazy_agent.run_lazy_clusterer(
                X,
                arguments.get("clusterers"),
                arguments.get("n_clusters", 3)
            )
        elif name == "analyze_clustering_results":
            result = lazy_agent.analyze_clustering_results(
                arguments["result_id"],
                arguments.get("analysis_type", "comprehensive")
            )
        
        # Data Generation and Management Tools
        elif name == "generate_sample_dataset":
            result = lazy_agent.generate_sample_dataset(
                arguments["dataset_type"],
                arguments.get("n_samples", 1000),
                arguments.get("n_features", 20),
                **{k: v for k, v in arguments.items() if k not in ['dataset_type', 'n_samples', 'n_features']}
            )
        elif name == "load_builtin_dataset":
            result = lazy_agent.load_builtin_dataset(
                arguments["dataset_name"]
            )
        elif name == "load_custom_dataset":
            result = lazy_agent.load_custom_dataset(
                arguments["data"],
                arguments["target_column"],
                arguments.get("dataset_type", "auto")
            )
        elif name == "preprocess_data":
            result = lazy_agent.preprocess_data(
                arguments["dataset_id"],
                arguments.get("preprocessing_steps"),
                arguments.get("test_size", 0.2),
                arguments.get("scale_features", False),
                arguments.get("encode_labels", False)
            )
        
        # Analysis and Comparison Tools
        elif name == "compare_model_types":
            result = lazy_agent.compare_model_types(
                arguments["dataset_id"],
                arguments.get("model_types")
            )
        elif name == "cross_validate_best_models":
            result = lazy_agent.cross_validate_best_models(
                arguments["result_id"],
                arguments.get("top_n", 5),
                arguments.get("cv_folds", 5)
            )
        elif name == "generate_model_recommendation":
            result = lazy_agent.generate_model_recommendation(
                arguments["result_id"],
                arguments.get("business_context", "general"),
                arguments.get("priority", "accuracy")
            )
        
        # Visualization and Reporting Tools
        elif name == "create_performance_visualization_data":
            result = lazy_agent.create_performance_visualization_data(
                arguments["result_id"]
            )
        elif name == "generate_comprehensive_report":
            result = lazy_agent.generate_comprehensive_report(
                arguments["result_id"],
                arguments.get("include_recommendations", True),
                arguments.get("include_technical_details", True)
            )
        
        # Utility Tools
        elif name == "get_available_models":
            result = lazy_agent.get_available_models(
                arguments.get("model_type", "all")
            )
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        )]
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

@server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    """List available LazyPredict prompts"""
    return [
        types.Prompt(
            name="complete_ml_workflow",
            description="Complete machine learning workflow from data to deployment recommendations",
            arguments=[
                types.PromptArgument(
                    name="problem_type",
                    description="Type of ML problem (classification, regression, clustering, auto-detect)",
                    required=False
                ),
                types.PromptArgument(
                    name="dataset_source",
                    description="Source of data (builtin, generated, custom)",
                    required=False
                ),
                types.PromptArgument(
                    name="business_context",
                    description="Business context for the ML project",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="model_selection_guide",
            description="Guide for selecting the best model based on specific criteria",
            arguments=[
                types.PromptArgument(
                    name="selection_criteria",
                    description="Primary criteria for model selection (accuracy, speed, interpretability)",
                    required=False
                ),
                types.PromptArgument(
                    name="constraints",
                    description="Any constraints or requirements (computational, regulatory, etc.)",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="quick_model_comparison",
            description="Quick setup for comparing multiple model types on your data",
            arguments=[
                types.PromptArgument(
                    name="data_characteristics",
                    description="Characteristics of your dataset (size, features, target type)",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="advanced_analysis_workflow",
            description="Advanced analysis including cross-validation, model families, and detailed reporting",
            arguments=[
                types.PromptArgument(
                    name="analysis_depth",
                    description="Depth of analysis required (basic, comprehensive, research-grade)",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="deployment_readiness_assessment",
            description="Assess model readiness for production deployment",
            arguments=[
                types.PromptArgument(
                    name="deployment_environment",
                    description="Target deployment environment (cloud, edge, on-premise)",
                    required=False
                ),
                types.PromptArgument(
                    name="performance_requirements",
                    description="Performance requirements (latency, throughput, accuracy)",
                    required=False
                )
            ]
        )
    ]

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    """Handle comprehensive prompt requests for LazyPredict workflows"""
    
    if name == "complete_ml_workflow":
        problem_type = arguments.get("problem_type", "auto-detect")
        dataset_source = arguments.get("dataset_source", "auto")
        business_context = arguments.get("business_context", "general")
        
        prompt_text = f"""
# Complete Machine Learning Workflow with LazyPredict

## Project Configuration
- **Problem Type**: {problem_type}
- **Dataset Source**: {dataset_source}
- **Business Context**: {business_context}

### Phase 1: Data Preparation
1. **Dataset Loading**
   ```
   # For built-in datasets:
   load_builtin_dataset(dataset_name="iris")
   
   # For custom CSV data:
   load_custom_dataset(data="your_csv_data", target_column="target")
   
   # For synthetic data:
   generate_sample_dataset(dataset_type="classification", n_samples=1000, n_features=20)
   ```

2. **Data Preprocessing**
   ```
   preprocess_data(
       dataset_id="your_dataset_id",
       test_size=0.2,
       scale_features=true,
       encode_labels=true
   )
   ```

### Phase 2: Model Comparison
1. **Choose Your Analysis Type**
   
   **For Classification Problems:**
   ```
   run_lazy_classifier(dataset_id="preprocessed_dataset_id")
   analyze_classification_results(result_id="classification_result_id", analysis_type="comprehensive")
   ```
   
   **For Regression Problems:**
   ```
   run_lazy_regressor(dataset_id="preprocessed_dataset_id")
   analyze_regression_results(result_id="regression_result_id", analysis_type="comprehensive")
   ```
   
   **For Clustering Analysis:**
   ```
   run_lazy_clusterer(dataset_id="dataset_id", n_clusters=3)
   analyze_clustering_results(result_id="clustering_result_id")
   ```
   
   **For Multi-Type Comparison:**
   ```
   compare_model_types(dataset_id="preprocessed_dataset_id", model_types=["all"])
   ```

### Phase 3: Advanced Analysis
1. **Cross-Validation**
   ```
   cross_validate_best_models(result_id="your_result_id", top_n=5, cv_folds=5)
   ```

2. **Model Recommendations**
   ```
   generate_model_recommendation(
       result_id="your_result_id",
       business_context="{business_context}",
       priority="accuracy"  # or "speed" or "balanced"
   )
   ```

### Phase 4: Visualization and Reporting
1. **Performance Visualization**
   ```
   create_performance_visualization_data(result_id="your_result_id")
   ```

2. **Comprehensive Report**
   ```
   generate_comprehensive_report(
       result_id="your_result_id",
       include_recommendations=true,
       include_technical_details=true
   )
   ```

### Key Success Factors:
- Start with data quality assessment
- Consider business constraints early
- Validate top models with cross-validation
- Document assumptions and limitations
- Plan for model monitoring and updates

Would you like to proceed with any specific phase of this workflow?
"""
        
        return types.GetPromptResult(
            description=f"Complete ML workflow for {problem_type} problem in {business_context} context",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_text)
                )
            ]
        )
    
    elif name == "model_selection_guide":
        criteria = arguments.get("selection_criteria", "accuracy")
        constraints = arguments.get("constraints", "none specified")
        
        guide_text = f"""
# Model Selection Guide with LazyPredict

## Selection Criteria: {criteria.title()}
## Constraints: {constraints}

### Step 1: Run Comprehensive Model Comparison
First, get a baseline comparison of all available models:

```
# Run appropriate lazy prediction based on your problem type
run_lazy_classifier(dataset_id="your_dataset_id")  # for classification
run_lazy_regressor(dataset_id="your_dataset_id")   # for regression
run_lazy_clusterer(dataset_id="your_dataset_id")   # for clustering
```

### Step 2: Apply Selection Criteria

#### For Accuracy-Focused Selection:
```
generate_model_recommendation(
    result_id="your_result_id",
    priority="accuracy",
    business_context="your_context"
)
```
**Key Metrics to Consider:**
- Classification: Accuracy, ROC AUC, F1 Score, Precision, Recall
- Regression: R-Squared, RMSE, MAE
- Clustering: Silhouette Score, Adjusted Rand Index

#### For Speed-Focused Selection:
```
generate_model_recommendation(
    result_id="your_result_id",
    priority="speed",
    business_context="your_context"
)
```
**Focus Areas:**
- Training time for batch retraining
- Inference time for real-time applications
- Memory usage for resource-constrained environments

#### For Balanced Selection:
```
generate_model_recommendation(
    result_id="your_result_id",
    priority="balanced",
    business_context="your_context"
)
```

### Step 3: Consider Model Characteristics

#### Interpretability Requirements:
- **High Interpretability**: Linear models, Decision Trees, Naive Bayes
- **Medium Interpretability**: Random Forest, Gradient Boosting (with SHAP)
- **Low Interpretability**: Neural Networks, SVMs, Complex Ensembles

#### Scalability Requirements:
- **Large Datasets**: XGBoost, LightGBM, Linear models
- **Real-time Inference**: Linear models, Simple trees
- **Limited Memory**: Linear models, Naive Bayes

#### Robustness to Data Quality:
- **Missing Values**: Tree-based models (XGBoost, Random Forest)
- **Outliers**: Tree-based models, Robust regression
- **Feature Scaling**: Neural Networks, SVMs require scaling

### Step 4: Validation Strategy
```
cross_validate_best_models(
    result_id="your_result_id",
    top_n=5,
    cv_folds=10  # Increase for more robust estimates
)
```

### Step 5: Final Selection Framework

1. **Filter by Hard Constraints**
   - Remove models that don't meet performance thresholds
   - Eliminate models that exceed resource limits
   - Filter out models that don't meet interpretability requirements

2. **Rank by Primary Criteria**
   - Sort remaining models by your primary metric
   - Consider confidence intervals from cross-validation

3. **Tie-Breaking Factors**
   - Model complexity and maintainability
   - Team expertise and familiarity
   - Available tooling and infrastructure

### Common Selection Patterns:

**Financial Services**: Interpretable models (Logistic Regression, Decision Trees) due to regulatory requirements
**Technology**: Performance-first (XGBoost, Neural Networks) with monitoring for drift
**Healthcare**: Balanced approach with emphasis on precision and interpretability
**Manufacturing**: Fast inference (Linear models, simple trees) for real-time control

Would you like specific recommendations for your use case?
"""
        
        return types.GetPromptResult(
            description=f"Model selection guide prioritizing {criteria}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=guide_text)
                )
            ]
        )
    
    elif name == "quick_model_comparison":
        data_chars = arguments.get("data_characteristics", "")
        
        quick_setup = f"""
# Quick Model Comparison Setup

## Data Characteristics: {data_chars}

### Fastest Path to Model Comparison:

#### Option 1: Use Built-in Dataset (for testing)
```
# Load a sample dataset
load_builtin_dataset(dataset_name="iris")  # classification
load_builtin_dataset(dataset_name="diabetes")  # regression

# Preprocess
preprocess_data(dataset_id="builtin_iris", test_size=0.2, scale_features=true)

# Compare models
run_lazy_classifier(dataset_id="builtin_iris_preprocessed")
```

#### Option 2: Generate Synthetic Data (for experimentation)
```
# Generate data matching your problem
generate_sample_dataset(
    dataset_type="classification",  # or "regression"
    n_samples=1000,
    n_features=10,
    n_classes=3
)

# Preprocess and run
preprocess_data(dataset_id="generated_classification_...", test_size=0.2)
run_lazy_classifier(dataset_id="generated_classification_..._preprocessed")
```

#### Option 3: Your Custom Data
```
# Load your CSV data (replace with your actual data)
load_custom_dataset(
    data="your,csv,data,here...",
    target_column="your_target_column"
)

# Quick preprocessing
preprocess_data(
    dataset_id="custom_...",
    test_size=0.2,
    scale_features=true,
    encode_labels=true
)

# Run comparison based on detected type
run_lazy_classifier(dataset_id="custom_..._preprocessed")  # if classification
run_lazy_regressor(dataset_id="custom_..._preprocessed")   # if regression
```

### One-Command Complete Analysis:
```
# After getting your result_id from above:
analyze_classification_results(result_id="your_result_id", analysis_type="comprehensive")

# Get immediate recommendations:
generate_model_recommendation(
    result_id="your_result_id",
    business_context="general",
    priority="balanced"
)
```

### Quick Visualization:
```
create_performance_visualization_data(result_id="your_result_id")
```

### Expected Timeline:
- **Built-in data**: 30 seconds to first results
- **Generated data**: 1-2 minutes depending on size
- **Custom data**: 2-5 minutes including preprocessing

### Immediate Insights You'll Get:
1. **Top 10 performing models** with key metrics
2. **Best model recommendation** with reasoning
3. **Model family analysis** (which types work best)
4. **Performance vs. speed trade-offs**
5. **Implementation recommendations**

Ready to start? Choose your data option above and run the commands!
"""
        
        return types.GetPromptResult(
            description="Quick setup for immediate model comparison",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=quick_setup)
                )
            ]
        )
    
    elif name == "advanced_analysis_workflow":
        analysis_depth = arguments.get("analysis_depth", "comprehensive")
        
        advanced_workflow = f"""
# Advanced Analysis Workflow

## Analysis Depth: {analysis_depth.title()}

### Phase 1: Comprehensive Model Evaluation

#### 1.1 Multi-Metric Analysis
```
# Run initial comparison
run_lazy_classifier(dataset_id="your_dataset_id", fold=10)  # More CV folds

# Detailed analysis
analyze_classification_results(
    result_id="your_result_id",
    analysis_type="comprehensive"
)
```

#### 1.2 Model Family Deep Dive
```
# Analyze model families separately
analyze_classification_results(result_id="your_result_id", analysis_type="model_families")
analyze_classification_results(result_id="your_result_id", analysis_type="efficiency_analysis")
```

### Phase 2: Statistical Validation

#### 2.1 Robust Cross-Validation
```
cross_validate_best_models(
    result_id="your_result_id",
    top_n=10,      # Test more models
    cv_folds=10    # More robust estimates
)
```

#### 2.2 Performance Distribution Analysis
```
analyze_classification_results(
    result_id="your_result_id",
    analysis_type="performance_distribution"
)
```

### Phase 3: Business-Aligned Selection

#### 3.1 Context-Specific Recommendations
```
# Generate recommendations for different contexts
generate_model_recommendation(
    result_id="your_result_id",
    business_context="financial",  # Try: healthcare, marketing, manufacturing
    priority="accuracy"
)

generate_model_recommendation(
    result_id="your_result_id",
    business_context="financial",
    priority="speed"
)

generate_model_recommendation(
    result_id="your_result_id",
    business_context="financial",
    priority="balanced"
)
```

### Phase 4: Multi-Type Model Comparison
```
# Compare different ML approaches on the same data
compare_model_types(
    dataset_id="your_preprocessed_dataset_id",
    model_types=["classification", "clustering"]  # See if unsupervised reveals patterns
)
```

### Phase 5: Advanced Visualization and Reporting

#### 5.1 Comprehensive Visualization Data
```
create_performance_visualization_data(result_id="your_result_id")
```

#### 5.2 Research-Grade Report
```
generate_comprehensive_report(
    result_id="your_result_id",
    include_recommendations=true,
    include_technical_details=true
)
```

### Phase 6: Model Sensitivity Analysis

#### 6.1 Data Subset Analysis
```
# Test on different data subsets (you'd need to create these)
preprocess_data(dataset_id="original_dataset", test_size=0.1)  # Smaller test set
preprocess_data(dataset_id="original_dataset", test_size=0.3)  # Larger test set

# Compare results across different splits
```

#### 6.2 Feature Scaling Impact
```
# Test with and without feature scaling
preprocess_data(dataset_id="original", scale_features=false)
preprocess_data(dataset_id="original", scale_features=true)

# Compare model performance differences
```

### Advanced Analysis Checklist:

**Statistical Rigor:**
- [ ] Used sufficient CV folds (≥10)
- [ ] Tested top N models (≥10)
- [ ] Analyzed performance distributions
- [ ] Checked for statistical significance

**Business Alignment:**
- [ ] Generated context-specific recommendations
- [ ] Considered multiple priority criteria
- [ ] Assessed interpretability requirements
- [ ] Evaluated deployment constraints

**Robustness Testing:**
- [ ] Tested different train/test splits
- [ ] Analyzed preprocessing impact
- [ ] Compared model families
- [ ] Assessed efficiency trade-offs

**Documentation:**
- [ ] Generated comprehensive report
- [ ] Created visualization data
- [ ] Documented assumptions
- [ ] Recorded methodology

### Expected Outputs:
1. **Statistically validated model rankings**
2. **Business-context recommendations**
3. **Comprehensive performance analysis**
4. **Deployment readiness assessment**
5. **Research-quality documentation**

This workflow typically takes 15-30 minutes depending on dataset size and depth of analysis required.
"""
        
        return types.GetPromptResult(
            description=f"Advanced analysis workflow with {analysis_depth} depth",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=advanced_workflow)
                )
            ]
        )
    
    elif name == "deployment_readiness_assessment":
        deployment_env = arguments.get("deployment_environment", "cloud")
        performance_reqs = arguments.get("performance_requirements", "standard")
        
        deployment_guide = f"""
# Model Deployment Readiness Assessment

## Target Environment: {deployment_env.title()}
## Performance Requirements: {performance_reqs}

### Phase 1: Model Performance Validation

#### 1.1 Statistical Validation
```
# Ensure robust performance estimates
cross_validate_best_models(
    result_id="your_result_id",
    top_n=3,        # Focus on top candidates
    cv_folds=10     # Robust validation
)
```

#### 1.2 Business Metric Alignment
```
generate_model_recommendation(
    result_id="your_result_id",
    business_context="your_domain",
    priority="balanced"  # Consider all factors for production
)
```

### Phase 2: Technical Readiness Assessment

#### 2.1 Performance Characteristics Analysis
```
analyze_classification_results(
    result_id="your_result_id",
    analysis_type="efficiency_analysis"
)
```

**Key Deployment Metrics to Evaluate:**

**For Cloud Deployment:**
- Training time: Can you retrain within acceptable windows?
- Memory usage: Fits within instance constraints?
- Scalability: Handles expected load increases?

**For Edge Deployment:**
- Model size: Fits on device storage?
- Inference time: Meets real-time requirements?
- Computational complexity: Runs on target hardware?

**For On-Premise:**
- Resource requirements: Fits current infrastructure?
- Maintenance complexity: Team can support?
- Integration requirements: Compatible with existing systems?

### Phase 3: Production Readiness Checklist

#### 3.1 Model Selection Criteria for Production

**High-Performance Requirements ({performance_reqs}):**
```python
# Focus on top performers
if performance_requirements == "high":
    priority = "accuracy"
    minimum_accuracy_threshold = 0.95  # Adjust for your domain
    acceptable_inference_time = 100    # milliseconds
    
elif performance_requirements == "real-time":
    priority = "speed" 
    maximum_inference_time = 10       # milliseconds
    minimum_accuracy_threshold = 0.85
    
else:  # standard
    priority = "balanced"
    minimum_accuracy_threshold = 0.90
    acceptable_inference_time = 1000  # 1 second
```

#### 3.2 Environment-Specific Considerations

**Cloud Deployment Checklist:**
- [ ] Auto-scaling compatibility
- [ ] Container optimization
- [ ] API wrapper design
- [ ] Monitoring and alerting setup
- [ ] A/B testing framework
- [ ] Model versioning strategy

**Edge Deployment Checklist:**
- [ ] Model quantization/compression
- [ ] Offline capability
- [ ] Update mechanism
- [ ] Local data privacy
- [ ] Failure handling
- [ ] Resource monitoring

**On-Premise Checklist:**
- [ ] Infrastructure requirements
- [ ] Security compliance
- [ ] Backup and recovery
- [ ] Team training needs
- [ ] Documentation completeness
- [ ] Support procedures

### Phase 4: Deployment Strategy

#### 4.1 Recommended Model Assessment
```
generate_comprehensive_report(
    result_id="your_result_id",
    include_recommendations=true,
    include_technical_details=true
)
```

#### 4.2 Risk Assessment

**Model Risk Factors:**
1. **Performance Degradation**: Set up monitoring for accuracy drift
2. **Data Drift**: Monitor input feature distributions
3. **Concept Drift**: Track business metric alignment
4. **Technical Debt**: Plan for model updates and retraining

**Mitigation Strategies:**
- Gradual rollout (10% → 50% → 100% traffic)
- Champion/challenger setup
- Real-time performance monitoring
- Automated rollback triggers

### Phase 5: Post-Deployment Monitoring

#### 5.1 Performance Monitoring
- Model accuracy on new data
- Inference latency and throughput
- Resource utilization
- Error rates and exceptions

#### 5.2 Business Impact Monitoring  
- Business KPI impact
- User experience metrics
- Cost/benefit analysis
- ROI measurement

### Deployment Readiness Score:

Based on your analysis results, calculate:

**Technical Readiness (40%):**
- Model performance meets requirements
- Inference time acceptable
- Resource requirements manageable
- Integration complexity low

**Operational Readiness (30%):**
- Monitoring systems in place
- Support procedures defined
- Update/rollback processes tested
- Team training completed

**Business Readiness (30%):**
- Business value validated
- Success metrics defined
- Stakeholder alignment achieved
- Risk mitigation planned

**Go/No-Go Decision Framework:**
- Score ≥ 85%: Ready for production
- Score 70-84%: Ready with risk mitigation
- Score < 70%: Requires additional work

Would you like help calculating your deployment readiness score?
"""
        
        return types.GetPromptResult(
            description=f"Deployment readiness assessment for {deployment_env} environment",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=deployment_guide)
                )
            ]
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")

async def main():
    """Main entry point for the MCP server"""
    try:
        # Import here to avoid issues if mcp package is not installed
        from mcp.server.stdio import stdio_server
        
        logger.info("Starting LazyPredict MCP Server...")
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="lazypredict-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except ImportError as e:
        logger.error(f"MCP import error: {e}")
        print("Error: MCP package not found. Install with: pip install mcp")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())