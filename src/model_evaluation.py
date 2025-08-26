"""
Credit Score Classification - Model Evaluation Module
Updated for project structure with trained models evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CreditScoreModelEvaluator:
    """Kredi Skoru Model Değerlendirici - Updated for project workflow"""
    
    def __init__(self, models_dir="../models/"):
        self.models_dir = models_dir
        self.class_names = ['Poor', 'Standard', 'Good']
        self.class_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        self.models = {}
        self.label_encoder = None
        self.cv_results = {}
        
    def load_trained_models(self):
        """Eğitilmiş modelleri yükle"""
        print("🔄 Eğitilmiş modeller yükleniyor...")
        
        # Label encoder'ı yükle
        encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ Label encoder yüklendi")
        else:
            print("⚠️ Label encoder bulunamadı!")
            
        # CV sonuçlarını yükle
        cv_path = os.path.join(self.models_dir, "cv_results.pkl")
        if os.path.exists(cv_path):
            with open(cv_path, 'rb') as f:
                self.cv_results = pickle.load(f)
            print("✅ CV sonuçları yüklendi")
        
        # Model dosyalarını tanımla
        model_files = {
            'RandomForest': 'randomforest_model.pkl',
            'LightGBM': 'lightgbm_model.pkl', 
            'XGBoost': 'xgboost_model.pkl',
            'LogisticRegression': 'logisticregression_model.pkl',
            'Ensemble': 'ensemble_model.pkl'
        }
        
        # Modelleri yükle
        loaded_count = 0
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ {model_name} yüklendi")
                    loaded_count += 1
                except Exception as e:
                    print(f"❌ {model_name} yükleme hatası: {str(e)}")
            else:
                print(f"⚠️ {model_name} dosyası bulunamadı: {filename}")
        
        print(f"\n📊 Toplam {loaded_count} model başarıyla yüklendi")
        return self.models
    
    def prepare_test_data(self, X_test, y_test):
        """Test verilerini hazırla"""
        print("\n🔄 Test verileri hazırlanıyor...")
        
        X_test_processed = X_test.copy()
        
        # Kategorik sütunları encode et
        categorical_cols = X_test_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_test_processed[col] = le.fit_transform(X_test_processed[col].astype(str))
        
        # Target variable'ı encode et
        if self.label_encoder is not None:
            if isinstance(y_test.iloc[0], str):
                y_test_encoded = self.label_encoder.transform(y_test)
            else:
                y_test_encoded = y_test.values
        else:
            # Eğer label encoder yoksa manuel mapping
            score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
            if isinstance(y_test.iloc[0], str):
                y_test_encoded = y_test.map(score_mapping).values
            else:
                y_test_encoded = y_test.values
        
        print(f"✅ Test verisi hazır - Shape: {X_test_processed.shape}")
        return X_test_processed, y_test_encoded
    
    def evaluate_single_model(self, model_name, model, X_test, y_test):
        """Tek model değerlendirmesi"""
        print(f"\n📊 {model_name} değerlendiriliyor...")
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                print(f"⚠️ {model_name} için probability tahminleri alınamadı")
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Sınıf bazında metrikler
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        # ROC-AUC hesaplama
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"⚠️ ROC-AUC hesaplanamadı: {str(e)}")
        
        # Sonuçları yazdır
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"🎯 F1-Score (Macro): {f1_macro:.4f}")
        print(f"🎯 F1-Score (Weighted): {f1_weighted:.4f}")
        if roc_auc:
            print(f"📈 ROC-AUC (Macro): {roc_auc:.4f}")
        
        # CV sonuçları varsa göster
        if model_name in self.cv_results:
            cv_mean = self.cv_results[model_name]['mean']
            cv_std = self.cv_results[model_name]['std']
            print(f"🔄 CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1,
            'support': support,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_score_mean': self.cv_results.get(model_name, {}).get('mean', None),
            'cv_score_std': self.cv_results.get(model_name, {}).get('std', None)
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """Tüm modelleri değerlendir"""
        print("="*60)
        print("🚀 TÜM MODELLERİN DEĞERLENDİRİLMESİ BAŞLADI")
        print("="*60)
        
        if not self.models:
            print("❌ Hiç model yüklenmedi! Önce load_trained_models() çalıştırın.")
            return {}
        
        # Test verilerini hazırla
        X_test_processed, y_test_encoded = self.prepare_test_data(X_test, y_test)
        
        # Her modeli değerlendir
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                results = self.evaluate_single_model(model_name, model, X_test_processed, y_test_encoded)
                evaluation_results[model_name] = results
            except Exception as e:
                print(f"❌ {model_name} değerlendirme hatası: {str(e)}")
        
        print("\n" + "="*60)
        print("✅ TÜM MODELLER DEĞERLENDİRİLDİ")
        print("="*60)
        
        return evaluation_results, X_test_processed, y_test_encoded
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", figsize=(8, 6)):
        """Confusion matrix görselleştir"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title(f'{model_name} - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized confusion matrix
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
        axes[1].set_title(f'{model_name} - Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        return cm, cm_normalized
    
    def plot_roc_curves(self, y_true, y_pred_proba, model_name="Model", figsize=(10, 8)):
        """ROC curves çizer (multiclass)"""
        if y_pred_proba is None:
            print(f"⚠️ {model_name} için probability tahminleri bulunamadı!")
            return
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]
        
        # ROC curve ve AUC hesapla
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        colors = ['blue', 'red', 'green']
        plt.figure(figsize=figsize)
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=2,
                label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", top_n=20, figsize=(10, 8)):
        """Feature importance görselleştir"""
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ {model_name} feature importance desteklemiyor!")
            return None
        
        # Feature importance al
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Top N features
        top_features = feature_importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def compare_models(self, evaluation_results, metric='f1_macro', figsize=(12, 6)):
        """Modelleri karşılaştır"""
        print(f"\n🏆 MODEL KARŞILAŞTIRMA ({metric.upper()})")
        print("="*60)
        
        # Sonuçları topla
        comparison_data = []
        for model_name, results in evaluation_results.items():
            if metric in results and results[metric] is not None:
                row = {
                    'Model': model_name,
                    f'Test_{metric}': results[metric]
                }
                
                # CV sonuçları varsa ekle
                if results.get('cv_score_mean') is not None:
                    row['CV_Mean'] = results['cv_score_mean']
                    row['CV_Std'] = results.get('cv_score_std', 0)
                
                comparison_data.append(row)
        
        # DataFrame oluştur
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(f'Test_{metric}', ascending=False)
        
        # Tablo göster
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Test scores
        bars1 = axes[0].bar(comparison_df['Model'], comparison_df[f'Test_{metric}'], 
                           color='skyblue', alpha=0.7)
        axes[0].set_title(f'Test {metric.upper()} Scores')
        axes[0].set_ylabel(f'Test {metric.upper()}')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # En iyi modeli vurgula
        best_idx = comparison_df[f'Test_{metric}'].idxmax()
        bars1[best_idx].set_color('gold')
        
        # CV vs Test karşılaştırması (eğer CV sonuçları varsa)
        if 'CV_Mean' in comparison_df.columns:
            x = np.arange(len(comparison_df))
            width = 0.35
            
            axes[1].bar(x - width/2, comparison_df['CV_Mean'], width, 
                       label='CV Mean', color='lightcoral', alpha=0.7)
            axes[1].bar(x + width/2, comparison_df[f'Test_{metric}'], width,
                       label=f'Test {metric}', color='skyblue', alpha=0.7)
            
            # Error bars for CV
            if 'CV_Std' in comparison_df.columns:
                axes[1].errorbar(x - width/2, comparison_df['CV_Mean'], 
                               yerr=comparison_df['CV_Std'], fmt='none', 
                               color='red', capsize=3)
            
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score')
            axes[1].set_title('CV vs Test Performance')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(comparison_df['Model'], rotation=45)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def create_comprehensive_report(self, evaluation_results, X_test, y_test, feature_names):
        """Kapsamlı değerlendirme raporu oluştur"""
        print("\n📋 KAPSAMLI DEĞERLENDİRME RAPORU OLUŞTURULUYOR...")
        print("="*80)
        
        # Her model için detaylı analiz
        for model_name, results in evaluation_results.items():
            print(f"\n🔍 {model_name.upper()} DETAYLI ANALİZ")
            print("-"*50)
            
            # Confusion Matrix
            self.plot_confusion_matrix(y_test, results['y_pred'], model_name)
            
            # ROC Curves
            if results['y_pred_proba'] is not None:
                self.plot_roc_curves(y_test, results['y_pred_proba'], model_name)
            
            # Feature Importance
            if hasattr(results['model'], 'feature_importances_'):
                self.plot_feature_importance(results['model'], feature_names, model_name)
            
            # Sınıf bazında detaylı rapor
            print(f"\n📊 {model_name} - Sınıf Bazında Performans:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}:")
                print(f"    Precision: {results['precision'][i]:.4f}")
                print(f"    Recall: {results['recall'][i]:.4f}")
                print(f"    F1-Score: {results['f1_scores'][i]:.4f}")
                print(f"    Support: {results['support'][i]}")
        
        # Model karşılaştırması
        comparison_df = self.compare_models(evaluation_results)
        
        # En iyi model
        best_model = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Test_f1_macro']
        
        print(f"\n🏆 EN İYİ MODEL: {best_model}")
        print(f"🎯 F1-Macro Score: {best_score:.4f}")
        
        return comparison_df, best_model
    
    def save_evaluation_results(self, evaluation_results, comparison_df, filepath="../models/evaluation_results.pkl"):
        """Değerlendirme sonuçlarını kaydet"""
        results_to_save = {
            'evaluation_results': {},
            'comparison_df': comparison_df,
            'timestamp': pd.Timestamp.now()
        }
        
        # Model objelerini çıkar (pickle edilemez olabilir)
        for model_name, results in evaluation_results.items():
            results_to_save['evaluation_results'][model_name] = {
                k: v for k, v in results.items() 
                if k not in ['model', 'y_pred', 'y_pred_proba']  # Bu büyük arrayler
            }
        
        # Kaydet
        with open(filepath, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"✅ Değerlendirme sonuçları kaydedildi: {filepath}")
    
    def generate_summary_report_text(self, evaluation_results, comparison_df):
        """Metin tabanlı özet rapor oluştur"""
        report_text = []
        report_text.append("="*80)
        report_text.append("CREDIT SCORE CLASSIFICATION - MODEL EVALUATION SUMMARY")
        report_text.append("="*80)
        report_text.append(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_text.append(f"Number of Models Evaluated: {len(evaluation_results)}")
        report_text.append("")
        
        # En iyi model
        best_model = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Test_f1_macro']
        report_text.append(f"🏆 BEST PERFORMING MODEL: {best_model}")
        report_text.append(f"🎯 F1-Macro Score: {best_score:.4f}")
        report_text.append("")
        
        # Tüm modellerin performansı
        report_text.append("📊 ALL MODELS PERFORMANCE:")
        report_text.append("-"*50)
        for _, row in comparison_df.iterrows():
            model_name = row['Model']
            test_score = row['Test_f1_macro']
            cv_score = row.get('CV_Mean', 'N/A')
            
            if cv_score != 'N/A':
                report_text.append(f"{model_name:<20}: Test={test_score:.4f}, CV={cv_score:.4f}")
            else:
                report_text.append(f"{model_name:<20}: Test={test_score:.4f}")
        
        report_text.append("")
        report_text.append("="*80)
        
        return "\n".join(report_text)

# Ana fonksiyon - Notebook'ta kullanım için
def run_complete_evaluation(X_test, y_test, models_dir="../models/"):

    
    print("🚀 COMPLETE MODEL EVALUATION PIPELINE BAŞLATILIYOR...")
    
    # Evaluator'ı başlat
    evaluator = CreditScoreModelEvaluator(models_dir=models_dir)
    
    # Modelleri yükle
    models = evaluator.load_trained_models()
    
    if not models:
        print("❌ Hiç model yüklenemedi! Pipeline durduruluyor.")
        return None, None, None
    
    # Tüm modelleri değerlendir
    evaluation_results, X_test_processed, y_test_encoded = evaluator.evaluate_all_models(X_test, y_test)
    
    # Kapsamlı rapor oluştur
    feature_names = X_test_processed.columns.tolist()
    comparison_df, best_model = evaluator.create_comprehensive_report(
        evaluation_results, y_test_encoded, feature_names
    )
    
    # Sonuçları kaydet
    evaluator.save_evaluation_results(evaluation_results, comparison_df)
    
    # Özet raporu yazdır
    summary_report = evaluator.generate_summary_report_text(evaluation_results, comparison_df)
    print("\n" + summary_report)
    
    return evaluator, evaluation_results, comparison_df
