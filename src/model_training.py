import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, X, y):
        """Veriyi model eğitimi için hazırla"""
        X_processed = X.copy()
        
        # Kategorik sütunları encode et
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Target variable'ı encode et
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_processed, y_encoded
    
    def cross_validate_model(self, model, X, y, groups, cv_folds=5):
        """Group-based cross validation"""
        group_kfold = GroupKFold(n_splits=cv_folds)
        cv_scores = cross_val_score(
            model, X, y, 
            cv=group_kfold, 
            groups=groups, 
            scoring='f1_weighted',
            n_jobs=-1
        )
        return cv_scores
    
    def train_random_forest(self, X, y, groups, cv_folds=5):
        """Random Forest model eğit"""
        print("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Cross validation
        cv_scores = self.cross_validate_model(model, X, y, groups, cv_folds)
        
        # Final model training
        model.fit(X, y)
        
        self.models['RandomForest'] = model
        self.cv_results['RandomForest'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"Random Forest CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return model
    
    def train_lightgbm(self, X, y, groups, cv_folds=5):
        """LightGBM model eğit"""
        print("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=self.random_state
        )
        
        # Cross validation
        cv_scores = self.cross_validate_model(model, X, y, groups, cv_folds)
        
        # Final model training
        model.fit(X, y)
        
        self.models['LightGBM'] = model
        self.cv_results['LightGBM'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"LightGBM CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return model
    
    def train_xgboost(self, X, y, groups, cv_folds=5):
        """XGBoost model eğit"""
        print("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        # Cross validation
        cv_scores = self.cross_validate_model(model, X, y, groups, cv_folds)
        
        # Final model training
        model.fit(X, y)
        
        self.models['XGBoost'] = model
        self.cv_results['XGBoost'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"XGBoost CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return model
    
    def train_logistic_regression(self, X, y, groups, cv_folds=5):
        """Logistic Regression model eğit"""
        print("Training Logistic Regression...")
        
        model = LogisticRegression(
            multi_class='ovr',
            max_iter=1000,
            random_state=self.random_state
        )
        
        # Cross validation
        cv_scores = self.cross_validate_model(model, X, y, groups, cv_folds)
        
        # Final model training
        model.fit(X, y)
        
        self.models['LogisticRegression'] = model
        self.cv_results['LogisticRegression'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"Logistic Regression CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return model
    
    def create_ensemble(self, X, y):
        """Ensemble model oluştur"""
        if len(self.models) < 2:
            print("Ensemble için yeterli model yok!")
            return None
        
        print("Creating Ensemble Model...")
        
        # En iyi 3 modeli al
        sorted_models = sorted(self.cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)
        top_models = [(name, self.models[name]) for name, _ in sorted_models[:3]]
        
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        ensemble.fit(X, y)
        
        self.models['Ensemble'] = ensemble
        
        print(f"Ensemble created with: {[name for name, _ in top_models]}")
        
        return ensemble
    
    def train_all(self, X, y, groups, cv_folds=5):
        """Tüm modelleri eğit"""
        print("=" * 50)
        print("STARTING MODEL TRAINING")
        print("=" * 50)
        
        # Veriyi hazırla
        X_processed, y_encoded = self.prepare_data(X, y)
        
        print(f"Data shape: {X_processed.shape}")
        print(f"Target classes: {len(np.unique(y_encoded))}")
        print("-" * 50)
        
        # Modelleri eğit
        self.train_random_forest(X_processed, y_encoded, groups, cv_folds)
        self.train_lightgbm(X_processed, y_encoded, groups, cv_folds)
        self.train_xgboost(X_processed, y_encoded, groups, cv_folds)
        self.train_logistic_regression(X_processed, y_encoded, groups, cv_folds)
        
        # Ensemble oluştur
        self.create_ensemble(X_processed, y_encoded)
        
        print("-" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        
        return self.models
    
    def save_models(self, model_dir="../models/"):
        """Modelleri kaydet"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print("Saving models...")
        
        # Modelleri kaydet
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name.lower()}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {model_name} -> {model_path}")
        
        # Label encoder'ı kaydet
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Saved: Label Encoder -> {encoder_path}")
        
        # CV sonuçlarını kaydet
        results_path = os.path.join(model_dir, "cv_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.cv_results, f)
        print(f"Saved: CV Results -> {results_path}")
        
        print("All models saved successfully!")
    
    def load_models(self, model_dir="../models/"):
        """Modelleri yükle"""
        print("Loading models...")
        
        # Label encoder'ı yükle
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # CV sonuçlarını yükle
        results_path = os.path.join(model_dir, "cv_results.pkl")
        with open(results_path, 'rb') as f:
            self.cv_results = pickle.load(f)
        
        # Modelleri yükle
        model_files = ['randomforest_model.pkl', 'lightgbm_model.pkl', 
                      'xgboost_model.pkl', 'logisticregression_model.pkl', 
                      'ensemble_model.pkl']
        
        model_names = ['RandomForest', 'LightGBM', 'XGBoost', 
                      'LogisticRegression', 'Ensemble']
        
        for model_file, model_name in zip(model_files, model_names):
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded: {model_name}")
        
        print("Models loaded successfully!")
    
    def get_cv_results(self):
        """CV sonuçlarını dataframe olarak döndür"""
        if not self.cv_results:
            return None
        
        results_df = pd.DataFrame({
            model: {
                'CV_Mean': results['mean'],
                'CV_Std': results['std']
            }
            for model, results in self.cv_results.items()
        }).T
        
        results_df = results_df.sort_values('CV_Mean', ascending=False)
        
        return results_df