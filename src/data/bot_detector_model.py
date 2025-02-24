"""
Module for training and evaluating machine learning models to detect bot reviews.
"""
import pandas as pd
import numpy as np
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BotDetectorModel:
    """
    A class for training and evaluating models to detect bot/suspicious reviews.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing feature data and where model will be saved
        """
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        
        # Create directories if they don't exist
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.feature_importances = None
        self.scaler = None
        
        logger.info(f"Initialized BotDetectorModel with data directory: {data_dir}")
        
    def load_features(self, movie_id: str) -> pd.DataFrame:
        """
        Load feature data for a specific movie.
        
        Args:
            movie_id: IMDB movie ID
            
        Returns:
            DataFrame containing features
        """
        if not movie_id.startswith('tt'):
            movie_id = f"tt{movie_id}"
            
        file_path = self.features_dir / f"{movie_id}_features.csv"
        
        if not file_path.exists():
            logger.error(f"Features file not found: {file_path}")
            return pd.DataFrame()
            
        features_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(features_df)} review features for movie {movie_id}")
        
        return features_df
    
    def combine_features(self, movie_ids: List[str]) -> pd.DataFrame:
        """
        Combine features from multiple movies for training.
        
        Args:
            movie_ids: List of IMDB movie IDs
            
        Returns:
            Combined DataFrame containing features from all movies
        """
        dfs = []
        
        for movie_id in movie_ids:
            df = self.load_features(movie_id)
            if not df.empty:
                # Add movie_id as a feature
                df['movie_id'] = movie_id
                dfs.append(df)
                
        if not dfs:
            logger.error("No feature data found for any of the specified movies")
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined features from {len(dfs)} movies, total {len(combined_df)} reviews")
        
        return combined_df
    
    def create_labeled_dataset(self, features_df: pd.DataFrame, 
                              label_rules: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create a labeled dataset for training using rule-based labeling.
    
        Args:
            features_df: DataFrame containing features
            label_rules: Optional dictionary of rules for labeling (if None, uses default rules)
        
        Returns:
            DataFrame with added 'is_suspicious' label
        """
    # Create a copy to avoid modifying the original
        df = features_df.copy()
    
        # Default rules for suspicious reviews
        if label_rules is None:
            label_rules = {
                'high_similarity': 0.8,  # Reviews with similarity above this threshold
                'extreme_sentiment': 0.9,  # Very extreme sentiment (positive or negative)
                'sentiment_mismatch': 0.7,  # Difference between title and content sentiment 
                'exclamation_ratio': 0.1,  # Reviews with many exclamation marks
                'uppercase_ratio': 0.3,  # Reviews with high percentage of uppercase
                'min_helpful_ratio': 0.1,  # Low helpful ratio threshold
                'extreme_rating_deviation': 3.0  # Large deviation from mean rating
            }
    
        # Apply rules to label reviews - FIXED VERSION WITH PROPER PARENTHESES
        conditions = (
            # Similarity-based conditions
            (df['max_similarity'] > label_rules['high_similarity']) | 
            (df['similar_reviews_count'] >= 2) |
        
            # Sentiment and text-based conditions
            ((abs(df['content_sentiment']) > label_rules['extreme_sentiment']) & 
            (df['vocabulary_richness'] < 0.4)) |
            (df['sentiment_diff'] > label_rules['sentiment_mismatch']) |
        
            # Suspicious patterns
            (df['exclamation_count'] / df['word_count'] > label_rules['exclamation_ratio']) |
            (df['uppercase_ratio'] > label_rules['uppercase_ratio']) |
        
            # Metadata-based conditions
            ((df['helpful_ratio'] < label_rules['min_helpful_ratio']) & 
            (df['rating_deviation'] > label_rules['extreme_rating_deviation'])) |
        
            # Burst detection
            (df['in_burst'] == True)
        )
    
        # Create label
        df['is_suspicious'] = conditions.astype(int)
    
        # Log statistics
        suspicious_count = df['is_suspicious'].sum()
        total_count = len(df)
        logger.info(f"Created labeled dataset: {suspicious_count} suspicious reviews out of {total_count} " +
                  f"({suspicious_count/total_count*100:.1f}%)")
    
        return df
    
    def prepare_training_data(self, labeled_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            labeled_df: DataFrame with features and 'is_suspicious' label
            
        Returns:
            Tuple of (X, y) with features and target variable
        """
        # Drop non-feature columns
        features_to_drop = ['review_id', 'movie_id', 'is_suspicious', 'in_burst']
        X = labeled_df.drop([col for col in features_to_drop if col in labeled_df.columns], axis=1)
        
        # Target variable
        y = labeled_df['is_suspicious']
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared training data with {X.shape[1]} features and {len(y)} samples")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'random_forest', 
                   tune_hyperparams: bool = True) -> object:
        """
        Train a model to detect suspicious reviews.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train ('random_forest', 'gradient_boosting', or 'logistic')
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        
        # Create pipeline with scaling
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif model_type == 'logistic':
            base_model = LogisticRegression(max_iter=1000, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Tune hyperparameters if requested
        if tune_hyperparams and len(y) > 20:  # Only tune if we have enough data
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_scaled, y)
            
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters: {best_params}")
            
            # Update model with best params
            self.model = grid_search.best_estimator_
        else:
            # Train with default parameters
            self.model = base_model
            self.model.fit(X_scaled, y)
        
        # Calculate feature importances for tree-based models
        if model_type in ['random_forest', 'gradient_boosting']:
            importances = self.model.feature_importances_
            self.feature_importances = dict(zip(X.columns, importances))
            
            # Log top features
            top_features = sorted(self.feature_importances.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
            logger.info("Top 10 features:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")
        
        logger.info("Model training complete")
        return self.model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("No model has been trained yet")
            return {}
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        cr = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        
        # Organize metrics
        metrics = {
            'accuracy': cr['accuracy'],
            'precision': cr['1']['precision'],
            'recall': cr['1']['recall'],
            'f1': cr['1']['f1-score'],
            'roc_auc': auc,
            'confusion_matrix': cm.tolist()
        }
        
        # Log results
        logger.info(f"Model evaluation results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation to get a more robust evaluation.
        """
        if self.model is None:
            logger.error("No model has been trained yet")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of folds based on dataset size and class distribution
        min_class_size = min(np.bincount(y))
        cv = min(cv, min_class_size)  # Ensure cv doesn't exceed smallest class size
        
        if cv < 2:
            logger.warning("Not enough samples for cross-validation. Skipping.")
            return {
                'accuracy': {'mean': None, 'std': None, 'values': []},
                'precision': {'mean': None, 'std': None, 'values': []},
                'recall': {'mean': None, 'std': None, 'values': []},
                'f1': {'mean': None, 'std': None, 'values': []},
                'roc_auc': {'mean': None, 'std': None, 'values': []}
            }
        
        # Perform cross-validation
        cv_accuracy = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='f1')
        cv_auc = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # Organize results
        cv_results = {
            'accuracy': {
                'mean': cv_accuracy.mean(),
                'std': cv_accuracy.std(),
                'values': cv_accuracy.tolist()
            },
            'precision': {
                'mean': cv_precision.mean(),
                'std': cv_precision.std(),
                'values': cv_precision.tolist()
            },
            'recall': {
                'mean': cv_recall.mean(),
                'std': cv_recall.std(),
                'values': cv_recall.tolist()
            },
            'f1': {
                'mean': cv_f1.mean(),
                'std': cv_f1.std(),
                'values': cv_f1.tolist()
            },
            'roc_auc': {
                'mean': cv_auc.mean(),
                'std': cv_auc.std(),
                'values': cv_auc.tolist()
            }
        }
        
        # Log results
        logger.info(f"Cross-validation results ({cv} folds):")
        logger.info(f"  Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        logger.info(f"  Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        logger.info(f"  Recall: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        logger.info(f"  F1 Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        logger.info(f"  ROC AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")
        
        return cv_results
    
    def save_model(self, model_name: str = 'bot_detector') -> None:
        """
        Save the trained model and scaler.
        
        Args:
            model_name: Name to save the model under
        """
        if self.model is None:
            logger.error("No model has been trained yet")
            return
            
        # Create full paths
        model_path = self.models_dir / f"{model_name}.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        feature_importances_path = self.models_dir / f"{model_name}_importances.json"
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save feature importances if available
        if self.feature_importances:
            with open(feature_importances_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_importances, f, indent=2)
                
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = 'bot_detector') -> None:
        """
        Load a previously trained model.
        
        Args:
            model_name: Name of the model to load
        """
        # Create full paths
        model_path = self.models_dir / f"{model_name}.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        feature_importances_path = self.models_dir / f"{model_name}_importances.json"
        
        if not model_path.exists() or not scaler_path.exists():
            logger.error(f"Model files not found at {model_path}")
            return
            
        # Load model and scaler
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load feature importances if available
        if feature_importances_path.exists():
            with open(feature_importances_path, 'r', encoding='utf-8') as f:
                self.feature_importances = json.load(f)
                
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict suspicious reviews using the trained model.
        
        Args:
            features_df: DataFrame containing review features
            
        Returns:
            DataFrame with original features plus prediction results
        """
        if self.model is None:
            logger.error("No model has been trained yet")
            return features_df
            
        # Create a copy to avoid modifying the original
        result_df = features_df.copy()
        
        # Drop non-feature columns if they exist
        features_to_drop = ['review_id', 'movie_id', 'is_suspicious', 'in_burst']
        X = result_df.drop([col for col in features_to_drop if col in result_df.columns], axis=1)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        result_df['prediction'] = self.model.predict(X_scaled)
        result_df['probability'] = self.model.predict_proba(X_scaled)[:, 1]
        
        num_suspicious = result_df['prediction'].sum()
        logger.info(f"Found {num_suspicious} suspicious reviews out of {len(result_df)} " +
                    f"({num_suspicious/len(result_df)*100:.1f}%)")
        
        return result_df
    
    def plot_feature_importance(self, top_n: int = 15) -> None:
        """
        Plot the feature importances.
        
        Args:
            top_n: Number of top features to display
        """
        if self.feature_importances is None:
            logger.error("No feature importances available")
            return
            
        # Sort features by importance
        sorted_features = sorted(self.feature_importances.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importances), y=list(features))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save plot
        output_path = self.results_dir / "feature_importance.png"
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
        
        # Show plot (if in interactive environment)
        plt.show()
    
    def plot_confusion_matrix(self, cm: List[List[int]]) -> None:
        """
        Plot a confusion matrix.
        
        Args:
            cm: Confusion matrix as a list of lists
        """
        cm_array = np.array(cm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Suspicious'],
                   yticklabels=['Legitimate', 'Suspicious'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save plot
        output_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")
        
        # Show plot (if in interactive environment)
        plt.show()
    
    def full_training_pipeline(self, movie_ids: List[str], 
                                 model_type: str = 'random_forest',
                                 tune_hyperparams: bool = True,
                                 test_size: float = 0.2,
                                 stratify: bool = True) -> Dict:
        """
        Run the full model training and evaluation pipeline.
    
        Args:
            movie_ids: List of movie IDs to use for training
            model_type: Type of model to train
            tune_hyperparams: Whether to tune hyperparameters
            test_size: Proportion of data to use for testing
            stratify: Whether to use stratified sampling for train/test split
        
        Returns:
            Dictionary with training results
        """
        # 1. Combine features from all movies
        all_features = self.combine_features(movie_ids)
        if all_features.empty:
            logger.error("Failed to load any features")
            return {}
        
        # 2. Create labeled dataset
        labeled_data = self.create_labeled_dataset(all_features)
    
        # 3. Split data into train and test
        X, y = self.prepare_training_data(labeled_data)
        
        # Validate test_size
        if not isinstance(test_size, float) or test_size <= 0 or test_size >= 1:
            logger.warning(f"Invalid test_size {test_size}, defaulting to 0.2")
            test_size = 0.2
        
        # Add debug logging
        logger.info(f"Dataset size: {len(y)}, test_size: {test_size}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Ensure minimum dataset size
        if len(y) < 4:  # Need at least 4 samples to split into train/test
            logger.error("Dataset too small for train/test split")
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None  # Disable stratify for now
        )
    
        # 4. Train model
        self.train_model(X_train, y_train, model_type=model_type, tune_hyperparams=tune_hyperparams)
    
        # 5. Evaluate on test set
        eval_metrics = self.evaluate_model(X_test, y_test)
    
        # 6. Cross-validate only if we have enough samples
        class_counts = np.bincount(y)
        if min(class_counts) >= 2:  # Need at least 2 samples per class
            cv_results = self.cross_validate(X, y)
        else:
            logger.warning("Not enough samples per class for cross-validation")
            cv_results = {
                'accuracy': {'mean': None, 'std': None, 'values': []},
                'precision': {'mean': None, 'std': None, 'values': []},
                'recall': {'mean': None, 'std': None, 'values': []},
                'f1': {'mean': None, 'std': None, 'values': []},
                'roc_auc': {'mean': None, 'std': None, 'values': []}
            }
    
        # 7. Plot feature importance
        if self.feature_importances:
            self.plot_feature_importance()
        
        # 8. Plot confusion matrix
        if 'confusion_matrix' in eval_metrics:
            self.plot_confusion_matrix(eval_metrics['confusion_matrix'])
        
        # 9. Save model
        self.save_model()
    
        # 10. Combine results
        results = {
            'test_metrics': eval_metrics,
            'cross_validation': cv_results,
            'data_stats': {
                'total_reviews': len(labeled_data),
                'suspicious_reviews': labeled_data['is_suspicious'].sum(),
                'legitimate_reviews': len(labeled_data) - labeled_data['is_suspicious'].sum(),
                'suspicious_percentage': labeled_data['is_suspicious'].mean() * 100
            }
        }
    
        # Save results to file
        results_path = self.results_dir / "training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy values to Python types for JSON serialization
            import json
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
                
            json.dump(results, f, indent=2, cls=NpEncoder)
    
        logger.info(f"Full training pipeline completed. Results saved to {results_path}")
        return results