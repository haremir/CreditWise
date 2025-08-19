import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class CreditFeatureEngineer:
    """
    Kredi skoru verisi için gelişmiş özellik mühendisliği sınıfı
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
    
    def check_and_fix_infinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infinity değerlerini kontrol et ve düzelt
        """
        df_fixed = df.copy()
        
        # Sayısal sütunlarda infinity değerlerini kontrol et
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        
        infinity_found = False
        for col in numeric_cols:
            inf_count = np.isinf(df_fixed[col]).sum()
            if inf_count > 0:
                infinity_found = True
                print(f"⚠️ {col} sütununda {inf_count} infinity değer bulundu")
                
                # Infinity değerleri medyan ile değiştir
                finite_values = df_fixed[col][np.isfinite(df_fixed[col])]
                if len(finite_values) > 0:
                    median_val = finite_values.median()
                else:
                    median_val = 0
                
                df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], median_val)
                print(f"  ✅ {col}: infinity değerler {median_val:.2f} ile değiştirildi")
        
        if not infinity_found:
            print("✅ Infinity değer bulunamadı")
        
        return df_fixed
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş özellikler oluştur
        """
        df_advanced = df.copy()
        
        print("=== GELİŞMİŞ ÖZELLİK OLUŞTURMA ===")
        
        # 1. Finansal Sağlamlık Skoru
        if all(col in df_advanced.columns for col in ['Annual_Income', 'Outstanding_Debt', 'Monthly_Balance']):
            # Sıfıra bölmeyi önle
            df_advanced['Financial_Health_Score'] = (
                (df_advanced['Annual_Income'] / 12 + df_advanced['Monthly_Balance']) / 
                (df_advanced['Outstanding_Debt'] + 1)
            )
            print("Finansal sağlamlık skoru oluşturuldu")
        
        # 2. Kredi Yoğunluğu
        if all(col in df_advanced.columns for col in ['Num_Credit_Card', 'Num_of_Loan', 'Num_Bank_Accounts']):
            # Sıfıra bölmeyi önle
            df_advanced['Credit_Density'] = (
                df_advanced['Num_Credit_Card'] + df_advanced['Num_of_Loan']
            ) / (df_advanced['Num_Bank_Accounts'] + 1)
            print("Kredi yoğunluğu oluşturuldu")
        
        # 3. Ödeme Performans Skoru
        if all(col in df_advanced.columns for col in ['Num_of_Delayed_Payment', 'Delay_from_due_date']):
            # Gecikme yoksa 100, varsa gecikme sayısı ve süresine göre azalan skor
            df_advanced['Payment_Performance_Score'] = np.where(
                (df_advanced['Num_of_Delayed_Payment'] == 0) | (df_advanced['Delay_from_due_date'] <= 0),
                100,
                100 / (1 + df_advanced['Num_of_Delayed_Payment'] * np.log1p(np.abs(df_advanced['Delay_from_due_date'])))
            )
            print("Ödeme performans skoru oluşturuldu")
        
        # 4. Yaş-Gelir Etkileşimi
        if all(col in df_advanced.columns for col in ['Age', 'Annual_Income']):
            df_advanced['Age_Income_Interaction'] = df_advanced['Age'] * np.log1p(np.abs(df_advanced['Annual_Income']))
            print("Yaş-gelir etkileşimi oluşturuldu")
        
        # 5. Kredi Kullanım Efektifliği
        if all(col in df_advanced.columns for col in ['Credit_Utilization_Ratio', 'Num_Credit_Card']):
            # Sıfıra bölmeyi önle
            df_advanced['Credit_Efficiency'] = df_advanced['Credit_Utilization_Ratio'] / (df_advanced['Num_Credit_Card'] + 1)
            print("Kredi kullanım efektifliği oluşturuldu")
        
        # 6. Gelir Stabilitesi (Monthly_Inhand_Salary vs Annual_Income consistency)
        if all(col in df_advanced.columns for col in ['Monthly_Inhand_Salary', 'Annual_Income']):
            expected_monthly = df_advanced['Annual_Income'] / 12
            # Sıfıra bölmeyi önle
            df_advanced['Income_Consistency'] = 1 - np.abs(
                df_advanced['Monthly_Inhand_Salary'] - expected_monthly
            ) / (expected_monthly + 1)
            df_advanced['Income_Consistency'] = np.clip(df_advanced['Income_Consistency'], 0, 1)
            print("Gelir tutarlılığı oluşturuldu")
        
        # 7. Borç-Varlık Oranı
        if all(col in df_advanced.columns for col in ['Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']):
            total_assets = df_advanced['Amount_invested_monthly'] * 12 + df_advanced['Monthly_Balance']
            # Sıfıra bölmeyi önle
            df_advanced['Debt_to_Asset_Ratio'] = df_advanced['Outstanding_Debt'] / (total_assets + 1)
            print("Borç-varlık oranı oluşturuldu")
        
        # 8. Kredi Mix Skoru (Credit Mix'e göre sayısal skor)
        if 'Credit_Mix' in df_advanced.columns:
            credit_mix_scores = {
                'Bad': 1,
                'Standard': 2,
                'Good': 3
            }
            df_advanced['Credit_Mix_Score'] = df_advanced['Credit_Mix'].map(credit_mix_scores).fillna(1)
            print("Kredi mix skoru oluşturuldu")
        
        # 9. Toplam Kredi Riski
        if all(col in df_advanced.columns for col in ['Credit_Utilization_Ratio', 'Num_of_Delayed_Payment', 'Outstanding_Debt']):
            df_advanced['Total_Credit_Risk'] = (
                df_advanced['Credit_Utilization_Ratio'] * 0.4 +
                df_advanced['Num_of_Delayed_Payment'] * 10 * 0.4 +
                np.log1p(np.abs(df_advanced['Outstanding_Debt'])) * 0.2
            )
            print("Toplam kredi riski oluşturuldu")
        
        # Infinity kontrolü
        df_advanced = self.check_and_fix_infinity(df_advanced)
        
        return df_advanced
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sayısal değişkenleri kategorik gruplara ayır
        """
        df_binned = df.copy()
        
        print("=== GRUPLAMA ÖZELLİKLERİ ===")
        
        # Gelir grupları
        if 'Annual_Income' in df_binned.columns:
            df_binned['Income_Group'] = pd.cut(
                df_binned['Annual_Income'],
                bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High', 'Premium']
            )
            income_dummies = pd.get_dummies(df_binned['Income_Group'], prefix='Income')
            df_binned = pd.concat([df_binned, income_dummies], axis=1)
            print("Gelir grupları oluşturuldu")
        
        # Kredi kullanım grupları
        if 'Credit_Utilization_Ratio' in df_binned.columns:
            df_binned['Credit_Usage_Group'] = pd.cut(
                df_binned['Credit_Utilization_Ratio'],
                bins=[0, 30, 50, 70, 90, 100],
                labels=['Low', 'Optimal', 'High', 'Very_High', 'Maxed']
            )
            usage_dummies = pd.get_dummies(df_binned['Credit_Usage_Group'], prefix='Credit_Usage')
            df_binned = pd.concat([df_binned, usage_dummies], axis=1)
            print("Kredi kullanım grupları oluşturuldu")
        
        # Yaş grupları (daha detaylı)
        if 'Age' in df_binned.columns:
            df_binned['Detailed_Age_Group'] = pd.cut(
                df_binned['Age'],
                bins=[0, 22, 30, 40, 50, 60, 100],
                labels=['Student', 'Young_Adult', 'Adult', 'Middle_Age', 'Pre_Senior', 'Senior']
            )
            age_dummies = pd.get_dummies(df_binned['Detailed_Age_Group'], prefix='Age')
            df_binned = pd.concat([df_binned, age_dummies], axis=1)
            print("Detaylı yaş grupları oluşturuldu")
        
        return df_binned
    
    def create_interaction_features(self, df: pd.DataFrame, important_features: list = None) -> pd.DataFrame:
        """
        Önemli özellikler arası etkileşim özellikleri oluştur
        """
        df_interaction = df.copy()
        
        print("=== ETKİLEŞİM ÖZELLİKLERİ ===")
        
        if important_features is None:
            # Varsayılan önemli özellikler
            important_features = [
                'Age', 'Annual_Income', 'Credit_Utilization_Ratio', 
                'Outstanding_Debt', 'Num_of_Delayed_Payment'
            ]
        
        # Mevcut özellikleri filtrele
        available_features = [f for f in important_features if f in df_interaction.columns]
        
        # İkili etkileşimler
        for i, feat1 in enumerate(available_features):
            for j, feat2 in enumerate(available_features[i+1:], i+1):
                interaction_name = f"{feat1}_x_{feat2}"
                df_interaction[interaction_name] = df_interaction[feat1] * df_interaction[feat2]
        
        print(f"{len(available_features)*(len(available_features)-1)//2} etkileşim özelliği oluşturuldu")
        
        # Infinity kontrolü
        df_interaction = self.check_and_fix_infinity(df_interaction)
        
        return df_interaction
    
    def create_polynomial_features(self, df: pd.DataFrame, features: list, degree: int = 2) -> pd.DataFrame:
        """
        Polinom özellikler oluştur
        """
        df_poly = df.copy()
        
        print(f"=== POLİNOM ÖZELLİKLER (Derece: {degree}) ===")
        
        # Mevcut özellikleri filtrele
        available_features = [f for f in features if f in df_poly.columns]
        
        if available_features:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = self.poly_features.fit_transform(df_poly[available_features])
            
            # Yeni özellik isimleri oluştur
            feature_names = self.poly_features.get_feature_names_out(available_features)
            
            # Orijinal özellikler dışındakileri ekle
            original_feature_count = len(available_features)
            new_features = feature_names[original_feature_count:]
            new_poly_data = poly_data[:, original_feature_count:]
            
            # DataFrame'e ekle
            for i, feature_name in enumerate(new_features):
                df_poly[f"poly_{feature_name}"] = new_poly_data[:, i]
            
            print(f"{len(new_features)} polinom özelliği oluşturuldu")
        
        # Infinity kontrolü
        df_poly = self.check_and_fix_infinity(df_poly)
        
        return df_poly
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      exclude_columns: list = None) -> pd.DataFrame:
        """
        Özellikleri ölçeklendir - Infinity kontrolü eklendi
        """
        df_scaled = df.copy()
        
        print(f"=== ÖZELLİK ÖLÇEKLENDİRME ({method.upper()}) ===")
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Önce infinity ve NaN değerleri temizle
        df_scaled = self.check_and_fix_infinity(df_scaled)
        
        # Sayısal sütunları bul
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        # Scaler seçimi
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Desteklenen yöntemler: 'standard', 'minmax', 'robust'")
        
        # Ölçeklendirme uygula
        if columns_to_scale:
            # Ölçeklendirmeden önce tekrar infinity kontrolü
            for col in columns_to_scale:
                if np.isinf(df_scaled[col]).any():
                    finite_values = df_scaled[col][np.isfinite(df_scaled[col])]
                    if len(finite_values) > 0:
                        median_val = finite_values.median()
                    else:
                        median_val = 0
                    df_scaled[col] = df_scaled[col].replace([np.inf, -np.inf], median_val)
                
                if df_scaled[col].isnull().any():
                    df_scaled[col].fillna(df_scaled[col].median(), inplace=True)
            
            df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
            print(f"{len(columns_to_scale)} özellik ölçeklendirildi")
        
        return df_scaled
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = 'f_classif', k: int = 50) -> pd.DataFrame:
        """
        En önemli özellikleri seç
        """
        print(f"=== ÖZELLİK SEÇİMİ ({method.upper()}, k={k}) ===")
        
        if target_column not in df.columns:
            print(f"Hedef sütun '{target_column}' bulunamadı")
            return df
        
        # Özellik ve hedef ayırma
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        
        X = df[numeric_features]
        y = df[target_column]
        
        # Infinity ve NaN temizleme
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Skorlama fonksiyonu seçimi
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Desteklenen yöntemler: 'f_classif', 'mutual_info'")
        
        # Özellik seçimi
        self.feature_selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_features)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Seçilen özellik isimleri
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        # Yeni DataFrame oluştur
        df_selected = df[selected_features + [target_column]].copy()
        
        # Kategorik sütunları da ekle
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            if col != target_column:
                df_selected[col] = df[col]
        
        print(f"{len(selected_features)} özellik seçildi")
        print("Seçilen özellikler:", selected_features[:10], "..." if len(selected_features) > 10 else "")
        
        return df_selected
    
    def apply_pca(self, df: pd.DataFrame, n_components: float = 0.95, 
                  exclude_columns: list = None) -> pd.DataFrame:
        """
        PCA uygula
        """
        df_pca = df.copy()
        
        print(f"=== PCA UYGULAMASI (n_components={n_components}) ===")
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Sayısal sütunları bul
        numeric_columns = df_pca.select_dtypes(include=[np.number]).columns
        pca_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if len(pca_columns) < 2:
            print("PCA için yeterli özellik yok")
            return df_pca
        
        # Infinity ve NaN temizleme
        for col in pca_columns:
            df_pca[col] = df_pca[col].replace([np.inf, -np.inf], np.nan)
            df_pca[col].fillna(df_pca[col].median(), inplace=True)
        
        # PCA uygula
        self.pca = PCA(n_components=n_components)
        pca_data = self.pca.fit_transform(df_pca[pca_columns])
        
        # PCA sütunları ekle
        for i in range(pca_data.shape[1]):
            df_pca[f'PCA_{i+1}'] = pca_data[:, i]
        
        print(f"PCA ile {len(pca_columns)} özellik -> {pca_data.shape[1]} bileşene dönüştürüldü")
        print(f"Korunan varyans: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return df_pca
    
    def feature_engineering_pipeline(self, df: pd.DataFrame, target_column: str = None,
                                   config: dict = None) -> pd.DataFrame:
        """
        Tam özellik mühendisliği pipeline'ı
        """
        if config is None:
            config = {
                'advanced_features': True,
                'binned_features': True,
                'interaction_features': False,  # Çok fazla özellik oluşturabileceği için varsayılan kapalı
                'polynomial_features': False,   # Çok fazla özellik oluşturabileceği için varsayılan kapalı
                'scaling': 'standard',
                'feature_selection': True,
                'pca': False,
                'k_best': 50
            }
        
        print("=== ÖZELLİK MÜHENDİSLİĞİ PIPELINE BAŞLADI ===\n")
        
        df_processed = df.copy()
        
        # 1. Gelişmiş özellikler
        if config.get('advanced_features', True):
            df_processed = self.create_advanced_features(df_processed)
        
        # 2. Gruplama özellikleri
        if config.get('binned_features', True):
            df_processed = self.create_binned_features(df_processed)
        
        # 3. Etkileşim özellikleri
        if config.get('interaction_features', False):
            important_features = ['Age', 'Annual_Income', 'Credit_Utilization_Ratio']
            df_processed = self.create_interaction_features(df_processed, important_features)
        
        # 4. Polinom özellikler
        if config.get('polynomial_features', False):
            poly_features = ['Age', 'Annual_Income', 'Credit_Utilization_Ratio']
            df_processed = self.create_polynomial_features(df_processed, poly_features, degree=2)
        
        # 5. Ölçeklendirme
        if config.get('scaling'):
            exclude_cols = [target_column] if target_column else []
            df_processed = self.scale_features(df_processed, 
                                             method=config['scaling'],
                                             exclude_columns=exclude_cols)
        
        # 6. Özellik seçimi
        if config.get('feature_selection', True) and target_column:
            k_best = config.get('k_best', 50)
            df_processed = self.select_features(df_processed, target_column, k=k_best)
        
        # 7. PCA
        if config.get('pca', False):
            exclude_cols = [target_column] if target_column else []
            df_processed = self.apply_pca(df_processed, exclude_columns=exclude_cols)
        
        # 8. Final temizlik
        df_processed = self.check_and_fix_infinity(df_processed)
        
        print("\n=== ÖZELLİK MÜHENDİSLİĞİ TAMAMLANDI ===")
        print(f"Başlangıç özellik sayısı: {df.shape[1]}")
        print(f"Final özellik sayısı: {df_processed.shape[1]}")
        print(f"Veri boyutu: {df_processed.shape}")
        
        return df_processed


# Kullanım örneği
if __name__ == "__main__":
    # Feature engineer'ı başlat
    fe = CreditFeatureEngineer()
    
    # Örnek veri (processed data'dan)
    try:
        df = pd.read_csv('data/processed/credit_data_processed.csv')
        
        # Pipeline konfigürasyonu
        config = {
            'advanced_features': True,
            'binned_features': True,
            'interaction_features': False,
            'polynomial_features': False,
            'scaling': 'standard',
            'feature_selection': True,
            'pca': False,
            'k_best': 50
        }
        
        # Pipeline'ı çalıştır
        df_engineered = fe.feature_engineering_pipeline(
            df, 
            target_column='Credit_Score',  # Hedef sütun varsa
            config=config
        )
        
        # Sonucu kaydet
        df_engineered.to_csv('data/processed/credit_data_engineered.csv', index=False)
        print(f"\nMühendislikten geçmiş veri kaydedildi!")
        
    except FileNotFoundError:
        print("İşlenmiş veri dosyası bulunamadı. Önce data_processing.py çalıştırın.")