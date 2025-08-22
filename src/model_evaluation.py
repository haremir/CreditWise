"""
Credit Score Classification - Model Evaluation Module
Model değerlendirme ve görselleştirme fonksiyonları
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupKFold
import shap
import warnings
warnings.filterwarnings('ignore')

class CreditScoreModelEvaluator:
    """Kredi Skoru Model Değerlendirici"""
    
    def __init__(self):
        self.class_names = ['Poor', 'Standard', 'Good']
        self.class_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Model performansını detaylı şekilde değerlendirir"""
        print(f"\n📊 {model_name} DEĞERLENDIRME RAPORU")
        print("="*50)
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"🎯 F1-Score (Macro): {f1_macro:.4f}")
        print(f"🎯 F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Sınıf bazında metrikler
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        print(f"\n📋 Sınıf Bazında Metrikler:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Classification Report
        print(f"\n📄 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # ROC-AUC (multiclass için)
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                print(f"📈 ROC-AUC (Macro): {roc_auc:.4f}")
            except:
                print("⚠️ ROC-AUC hesaplanamadı")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1,
            'support': support,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", figsize=(8, 6)):
        """Confusion matrix görselleştirir"""
        plt.figure(figsize=figsize)
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        
        # Confusion matrix çiz
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Normalized confusion matrix çiz
        plt.figure(figsize=figsize)
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} - Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm, cm_normalized
    
    def plot_roc_curves(self, y_true, y_pred_proba, model_name="Model", figsize=(12, 8)):
        """ROC curves çizer (multiclass)"""
        if y_pred_proba is None:
            print("⚠️ Probability tahminleri bulunamadı!")
            return
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]
        
        # ROC curve ve AUC hesapla
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot
        plt.figure(figsize=figsize)
        colors = ['blue', 'red', 'green']
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
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
        """Feature importance görselleştirir"""
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ {model_name} feature importance desteklemiyor!")
            return
        
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
    
    def shap_analysis(self, model, X_sample, feature_names, model_name="Model", max_display=20):
        """SHAP analizi yapar"""
        print(f"\n🧠 {model_name} SHAP ANALİZİ...")
        
        try:
            # Tree modeller için TreeExplainer
            if hasattr(model, 'tree_'):
                explainer = shap.TreeExplainer(model)
            # Ensemble modeller için
            elif hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
            # Diğerleri için
            else:
                explainer = shap.Explainer(model, X_sample[:100])
            
            # SHAP values hesapla
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):  # Multiclass
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                class_names=self.class_names, max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                max_display=max_display, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.show()
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                # Multiclass için tüm sınıfların ortalamasını al
                mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(0)
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            top_features = feature_importance.head(max_display)
            sns.barplot(data=top_features, x='shap_importance', y='feature', palette='viridis')
            plt.title(f'{model_name} - SHAP Feature Importance')
            plt.xlabel('Mean |SHAP value|')
            plt.tight_layout()
            plt.show()
            
            return shap_values, feature_importance
            
        except Exception as e:
            print(f"⚠️ SHAP analizi başarısız: {str(e)}")
            return None, None
    
    def compare_models(self, model_results, metric='f1_macro'):
        """Modelleri karşılaştırır"""
        print(f"\n🏆 MODEL KARŞILAŞTIRMA ({metric.upper()})")
        print("="*60)
        
        # Sonuçları topla
        comparison_data = []
        for model_name, results in model_results.items():
            if metric in results:
                comparison_data.append({
                    'Model': model_name,
                    metric: results[metric],
                    'std': results.get(f'{metric}_std', 0)
                })
        
        # DataFrame oluştur
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric, ascending=False)
        
        # Tablo göster
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Görselleştirme
        plt.figure(figsize=(12, 6))
        bars = plt.bar(comparison_df['Model'], comparison_df[metric], 
                      capsize=5, color='skyblue', alpha=0.7)
        
        # Error bars ekle
        if 'std' in comparison_df.columns:
            plt.errorbar(comparison_df['Model'], comparison_df[metric], 
                        yerr=comparison_df['std'], fmt='none', color='red', capsize=5)
        
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # En iyi modeli vurgula
        best_idx = comparison_df[metric].idxmax()
        bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def plot_learning_curves(self, model, X, y, groups, model_name="Model", cv_folds=5):
        """Learning curves çizer"""
        from sklearn.model_selection import learning_curve
        
        print(f"\n📈 {model_name} Learning Curves hesaplanıyor...")
        
        # GroupKFold kullan
        cv = GroupKFold(n_splits=cv_folds)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, groups=groups,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1_macro', n_jobs=-1
        )
        
        # Ortalama ve std hesapla
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        
        plt.title(f'{model_name} - Learning Curves')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1-Macro Score')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return train_sizes, train_scores, val_scores
    
    def plot_validation_curves(self, model, X, y, groups, param_name, param_range, model_name="Model"):
        """Validation curves çizer"""
        from sklearn.model_selection import validation_curve
        
        print(f"\n📊 {model_name} Validation Curves hesaplanıyor...")
        
        # GroupKFold kullan
        cv = GroupKFold(n_splits=5)
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, groups=groups, scoring='f1_macro', n_jobs=-1
        )
        
        # Ortalama ve std hesapla
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        
        plt.title(f'{model_name} - Validation Curves ({param_name})')
        plt.xlabel(param_name)
        plt.ylabel('F1-Macro Score')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if isinstance(param_range[0], (int, float)) and len(param_range) > 5:
            plt.xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return train_scores, val_scores
    
    def create_evaluation_report(self, model_results, X_test, y_test, feature_names):
        """Kapsamlı değerlendirme raporu oluşturur"""
        print("\n📋 KAPSAMLI DEĞERLENDİRME RAPORU OLUŞTURULUYOR...")
        
        report = {
            'model_comparison': {},
            'detailed_results': {},
            'feature_importance': {}
        }
        
        # Her model için detaylı analiz
        for model_name, model_data in model_results.items():
            if 'model' in model_data:
                model = model_data['model']
                
                print(f"\n🔍 {model_name} analiz ediliyor...")
                
                # Model değerlendirme
                eval_results = self.evaluate_model(model, X_test, y_test, model_name)
                report['detailed_results'][model_name] = eval_results
                
                # Confusion matrix
                self.plot_confusion_matrix(y_test, eval_results['y_pred'], model_name)
                
                # ROC curves
                if eval_results['y_pred_proba'] is not None:
                    self.plot_roc_curves(y_test, eval_results['y_pred_proba'], model_name)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feat_imp = self.plot_feature_importance(model, feature_names, model_name)
                    report['feature_importance'][model_name] = feat_imp
                
                # Model comparison için temel metrikler
                report['model_comparison'][model_name] = {
                    'f1_macro': eval_results['f1_macro'],
                    'accuracy': eval_results['accuracy'],
                    'f1_weighted': eval_results['f1_weighted']
                }
        
        # Model karşılaştırma
        self.compare_models(report['model_comparison'], 'f1_macro')
        
        return report
    
    def save_evaluation_results(self, results, filepath):
        """Değerlendirme sonuçlarını kaydeder"""
        import pickle
        
        # Model objelerini çıkar (pickle edilemez)
        results_to_save = {}
        for key, value in results.items():
            if key != 'detailed_results':
                results_to_save[key] = value
            else:
                results_to_save[key] = {}
                for model_name, model_results in value.items():
                    results_to_save[key][model_name] = {
                        k: v for k, v in model_results.items() 
                        if not k.startswith('y_pred')  # Tahminleri de çıkar
                    }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"✅ Değerlendirme sonuçları kaydedildi: {filepath}")
    
    def generate_summary_report(self, model_results):
        """Özet rapor oluşturur"""
        print("\n📝 ÖZET RAPOR")
        print("="*80)
        
        # En iyi modeli bul
        best_score = 0
        best_model = None
        
        for model_name, results in model_results.items():
            if 'f1_macro' in results and results['f1_macro'] > best_score:
                best_score = results['f1_macro']
                best_model = model_name
        
        print(f"🏆 EN İYİ MODEL: {best_model}")
        print(f"🎯 F1-Macro Score: {best_score:.4f}")
        
        # Performance kategorileri
        performance_categories = {
            'Mükemmel (>0.85)': [],
            'İyi (0.75-0.85)': [],
            'Orta (0.65-0.75)': [],
            'Düşük (<0.65)': []
        }
        
        for model_name, results in model_results.items():
            if 'f1_macro' in results:
                score = results['f1_macro']
                if score > 0.85:
                    performance_categories['Mükemmel (>0.85)'].append((model_name, score))
                elif score > 0.75:
                    performance_categories['İyi (0.75-0.85)'].append((model_name, score))
                elif score > 0.65:
                    performance_categories['Orta (0.65-0.75)'].append((model_name, score))
                else:
                    performance_categories['Düşük (<0.65)'].append((model_name, score))
        
        # Kategorileri yazdır
        for category, models in performance_categories.items():
            if models:
                print(f"\n📊 {category}:")
                for model_name, score in sorted(models, key=lambda x: x[1], reverse=True):
                    print(f"   • {model_name}: {score:.4f}")
        
        return best_model, best_score