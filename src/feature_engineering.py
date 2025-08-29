import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class CreditFeatureEngineer:
    """
    Kredi skoru verisi iÃ§in MÄ°NÄ°MAL Ã¶zellik mÃ¼hendisliÄŸi sÄ±nÄ±fÄ±
    Overfitting'i Ã¶nleyici konservatif yaklaÅŸÄ±m
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        
        # Ä°ÅŸlem sÄ±rasÄ±nda takip edilecek temel Ã¶zellikler
        self.essential_features = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Credit_Utilization_Ratio',
            'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly',
            'Num_of_Delayed_Payment', 'Delay_from_due_date', 'Monthly_Balance',
            'Num_Credit_Card', 'Num_Bank_Accounts'
        ]
    
    def check_and_fix_infinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infinity deÄŸerlerini kontrol et ve dÃ¼zelt
        """
        df_fixed = df.copy()
        
        # SayÄ±sal sÃ¼tunlarda infinity deÄŸerlerini kontrol et
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        
        infinity_found = False
        for col in numeric_cols:
            inf_count = np.isinf(df_fixed[col]).sum()
            if inf_count > 0:
                infinity_found = True
                print(f"  {col} sÃ¼tununda {inf_count} infinity deÄŸer bulundu")
                
                # Infinity deÄŸerleri finite deÄŸerlerin medyanÄ± ile deÄŸiÅŸtir
                finite_values = df_fixed[col][np.isfinite(df_fixed[col])]
                if len(finite_values) > 0:
                    median_val = finite_values.median()
                    if pd.isna(median_val):
                        median_val = 0
                else:
                    median_val = 0
                
                df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], median_val)
                print(f"    {col}: infinity deÄŸerler {median_val:.2f} ile deÄŸiÅŸtirildi")
        
        if not infinity_found:
            print("Infinity deÄŸer bulunamadÄ±")
        
        # NaN deÄŸerleri de temizle
        nan_cols = df_fixed.isnull().sum()
        if nan_cols.sum() > 0:
            print("Kalan NaN deÄŸerler temizleniyor:")
            for col, count in nan_cols[nan_cols > 0].items():
                if col in numeric_cols:
                    median_val = df_fixed[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_fixed[col].fillna(median_val, inplace=True)
                    print(f"    {col}: {count} NaN deÄŸer {median_val:.2f} ile dolduruldu")
        
        return df_fixed
    
    def create_minimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SADECE 3-4 kritik Ã¶zellik oluÅŸtur
        """
        df_featured = df.copy()
        
        print("=== MÄ°NÄ°MAL Ã–ZELLÄ°K OLUÅžTURMA ===")
        
        feature_count = 0
        
        # 1. BorÃ§-Gelir OranÄ± (kritik risk gÃ¶stergesi)
        if all(col in df_featured.columns for col in ['Outstanding_Debt', 'Monthly_Income']):
            df_featured['Debt_Income_Ratio'] = df_featured['Outstanding_Debt'] / (df_featured['Monthly_Income'] + 1)
            feature_count += 1
            print("Debt_Income_Ratio oluÅŸturuldu")
        elif all(col in df_featured.columns for col in ['Outstanding_Debt', 'Annual_Income']):
            monthly_income = df_featured['Annual_Income'] / 12
            df_featured['Debt_Income_Ratio'] = df_featured['Outstanding_Debt'] / (monthly_income + 1)
            feature_count += 1
            print("Debt_Income_Ratio oluÅŸturuldu (Annual_Income'dan)")
        
        # 2. YaÅŸ-Gelir NormalleÅŸtirilmiÅŸ Skoru 
        if all(col in df_featured.columns for col in ['Age', 'Annual_Income']):
            # Log transformation ile aykÄ±rÄ± deÄŸerleri yumuÅŸat
            df_featured['Age_Income_Score'] = df_featured['Age'] * np.log1p(df_featured['Annual_Income'])
            feature_count += 1
            print("Age_Income_Score oluÅŸturuldu")
        
        # 3. Kredi YoÄŸunluÄŸu (kredi Ã¼rÃ¼nÃ¼ Ã§eÅŸitliliÄŸi)
        if all(col in df_featured.columns for col in ['Num_Credit_Card', 'Num_of_Loan', 'Num_Bank_Accounts']):
            total_credit_products = df_featured['Num_Credit_Card'] + df_featured['Num_of_Loan']
            df_featured['Credit_Product_Density'] = total_credit_products / (df_featured['Num_Bank_Accounts'] + 1)
            feature_count += 1
            print("Credit_Product_Density oluÅŸturuldu")
        
        # 4. Risk Skoru (gecikme + kullanÄ±m oranÄ± kombinasyonu)
        if all(col in df_featured.columns for col in ['Credit_Utilization_Ratio', 'Total_Delay_Score']):
            # Total_Delay_Score zaten veri iÅŸlemede oluÅŸturulmuÅŸtu
            df_featured['Combined_Risk_Score'] = (
                df_featured['Credit_Utilization_Ratio'] * 0.6 +
                np.minimum(df_featured['Total_Delay_Score'], 100) * 0.4
            )
            feature_count += 1
            print("Combined_Risk_Score oluÅŸturuldu")
        
        # Infinity kontrolÃ¼
        df_featured = self.check_and_fix_infinity(df_featured)
        
        print(f"Toplam {feature_count} minimal Ã¶zellik oluÅŸturuldu")
        
        return df_featured
    
    def scale_features_safe(self, df: pd.DataFrame, method: str = 'standard', 
                          exclude_columns: list = None) -> pd.DataFrame:
        """
        Ã–zellikleri gÃ¼venli ÅŸekilde Ã¶lÃ§eklendir - sÃ¼tun sÄ±ralamasÄ±nÄ± koruyarak
        """
        df_scaled = df.copy()
        
        print(f"=== GÃœVENLÄ° Ã–ZELLÄ°K Ã–LÃ‡EKLENDÄ°RME ({method.upper()}) ===")
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Ã–nce infinity ve NaN deÄŸerleri temizle
        df_scaled = self.check_and_fix_infinity(df_scaled)
        
        # ORÄ°JÄ°NAL SÃœTUN SIRALAMASINI SAKLA
        original_columns = df_scaled.columns.tolist()
        
        # SayÄ±sal sÃ¼tunlarÄ± bul
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        # Scaler seÃ§imi
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Desteklenen yÃ¶ntemler: 'standard', 'minmax', 'robust'")
        
        # Ã–lÃ§eklendirme uygula
        if columns_to_scale:
            # Son bir kez daha temizlik
            for col in columns_to_scale:
                # Infinity kontrolÃ¼
                if np.isinf(df_scaled[col]).any():
                    finite_values = df_scaled[col][np.isfinite(df_scaled[col])]
                    median_val = finite_values.median() if len(finite_values) > 0 else 0
                    df_scaled[col] = df_scaled[col].replace([np.inf, -np.inf], median_val)
                
                # NaN kontrolÃ¼
                if df_scaled[col].isnull().any():
                    df_scaled[col].fillna(df_scaled[col].median(), inplace=True)
            
            # Ã–lÃ§eklendirme uygula - SADECE BELÄ°RLÄ° SÃœTUNLARI
            try:
                scaled_data = self.scaler.fit_transform(df_scaled[columns_to_scale])
                
                # Ã–lÃ§eklendirilmiÅŸ deÄŸerleri GERÄ° YERLEÅžTÄ°R
                for i, col in enumerate(columns_to_scale):
                    df_scaled[col] = scaled_data[:, i]
                
                print(f"{len(columns_to_scale)} Ã¶zellik Ã¶lÃ§eklendirildi")
                
                # SÃœTUN SIRALAMASINI KORU
                df_scaled = df_scaled[original_columns]
                
            except Exception as e:
                print(f"Ã–lÃ§eklendirme hatasÄ±: {e}")
                # Hata durumunda orijinal deÄŸerleri koru
                return df
        
        return df_scaled
    
    def select_best_features(self, df: pd.DataFrame, target_column: str, 
                           method: str = 'f_classif', k: int = 25) -> pd.DataFrame:
        """
        En Ã¶nemli Ã¶zellikleri seÃ§ - konservatif k deÄŸeri
        """
        print(f"=== Ã–ZELLÄ°K SEÃ‡Ä°MÄ° ({method.upper()}, k={k}) ===")
        
        if target_column not in df.columns:
            print(f"Hedef sÃ¼tun '{target_column}' bulunamadÄ±")
            return df
        
        # Ã–zellik ve hedef ayÄ±rma
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        
        # Ã‡ok az Ã¶zellik varsa Ã¶zellik seÃ§imi atla
        if len(numeric_features) < 10:
            print("Yeterli sayÄ±sal Ã¶zellik yok, Ã¶zellik seÃ§imi atlanÄ±yor")
            return df
        
        X = df[numeric_features]
        y = df[target_column]
        
        # Temizlik
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Hedef deÄŸiÅŸkeni kontrol et (classification iÃ§in)
        if y.dtype == 'object':
            # String deÄŸerleri sayÄ±sal kodlara Ã§evir
            unique_values = y.unique()
            value_mapping = {val: idx for idx, val in enumerate(unique_values)}
            y = y.map(value_mapping)
        
        # Skorlama fonksiyonu seÃ§imi
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Desteklenen yÃ¶ntemler: 'f_classif', 'mutual_info'")
        
        # Ã–zellik seÃ§imi
        k_actual = min(k, len(numeric_features))
        self.feature_selector = SelectKBest(score_func=score_func, k=k_actual)
        
        try:
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # SeÃ§ilen Ã¶zellik isimleri
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            # Yeni DataFrame oluÅŸtur
            df_selected = df[selected_features + [target_column]].copy()
            
            # Kategorik sÃ¼tunlarÄ± da ekle (eÄŸer varsa)
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_columns:
                if col != target_column and col not in df_selected.columns:
                    df_selected[col] = df[col]
            
            print(f"{len(selected_features)} Ã¶zellik seÃ§ildi")
            if len(selected_features) <= 15:
                print("SeÃ§ilen Ã¶zellikler:", selected_features)
            else:
                print("SeÃ§ilen Ã¶zellikler (ilk 15):", selected_features[:15])
            
            return df_selected
            
        except Exception as e:
            print(f"Ã–zellik seÃ§imi hatasÄ±: {e}")
            return df
    
    def minimal_pipeline(self, df: pd.DataFrame, target_column: str = 'Credit_Score') -> pd.DataFrame:
        """
        MÄ°NÄ°MAL ve GÃœVENLÄ° Ã¶zellik mÃ¼hendisliÄŸi pipeline'Ä±
        """
        print("=== MÄ°NÄ°MAL Ã–ZELLÄ°K MÃœHENDÄ°SLÄ°ÄžÄ° PIPELINE ===\n")
        
        df_processed = df.copy()
        initial_features = df_processed.shape[1]
        
        print(f"BaÅŸlangÄ±Ã§ boyutu: {df_processed.shape}")
        
        # HEDEF SÃœTUNU Ä°LK BAÅžTA BELÄ°RLE VE KORU
        if target_column and target_column in df_processed.columns:
            target_data = df_processed[target_column].copy()
            print(f"Hedef sÃ¼tun '{target_column}' korundu")
        else:
            target_data = None
            print("Hedef sÃ¼tun belirtilmedi veya bulunamadÄ±")
        
        # 1. Minimal Ã¶zellik oluÅŸturma (sadece 3-4 tane)
        df_processed = self.create_minimal_features(df_processed)
        
        # 2. GÃ¼venli Ã¶lÃ§eklendirme - HEDEF VE KATEGORÄ°K SÃœTUNLARI HARIC TUT
        exclude_cols = [target_column] if target_column else []
        
        # Kategorik sÃ¼tunlarÄ± da hariÃ§ tut
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        exclude_cols.extend(categorical_cols)
        
        # ID sÃ¼tunlarÄ± da hariÃ§ tut
        id_columns = ['ID', 'Customer_ID', 'SSN', 'Name']
        for id_col in id_columns:
            if id_col in df_processed.columns and id_col not in exclude_cols:
                exclude_cols.append(id_col)
        
        df_processed = self.scale_features_safe(
            df_processed, 
            method='standard',
            exclude_columns=exclude_cols
        )
        
        # 3. Ã–zellik seÃ§imi (isteÄŸe baÄŸlÄ± - sadece Ã§ok fazla Ã¶zellik varsa)
        if df_processed.shape[1] > 35:  # 35'ten fazla Ã¶zellik varsa seÃ§im yap
            print("Ã‡ok fazla Ã¶zellik var, Ã¶zellik seÃ§imi yapÄ±lÄ±yor...")
            df_processed = self.select_best_features(
                df_processed, 
                target_column, 
                method='f_classif', 
                k=25
            )
        else:
            print(f"Ã–zellik sayÄ±sÄ± uygun ({df_processed.shape[1]}), seÃ§im yapÄ±lmadÄ±")
        
        # 4. Final temizlik
        df_processed = self.check_and_fix_infinity(df_processed)
        
        # 5. HEDEF SÃœTUNU SONA TAÅžI (standart ML formatÄ±)
        if target_column and target_column in df_processed.columns:
            cols = df_processed.columns.tolist()
            if target_column in cols:
                cols.remove(target_column)
                cols.append(target_column)  # Sona ekle
                df_processed = df_processed[cols]
                print(f"Hedef sÃ¼tun '{target_column}' son pozisyona taÅŸÄ±ndÄ±")
        
        final_features = df_processed.shape[1]
        
        print("\n=== MÄ°NÄ°MAL Ã–ZELLÄ°K MÃœHENDÄ°SLÄ°ÄžÄ° TAMAMLANDI ===")
        print(f"BaÅŸlangÄ±Ã§ Ã¶zellik sayÄ±sÄ±: {initial_features}")
        print(f"Final Ã¶zellik sayÄ±sÄ±: {final_features}")
        print(f"Ã–zellik deÄŸiÅŸimi: {final_features - initial_features:+d}")
        print(f"Veri boyutu: {df_processed.shape}")
        
        # SÃ¼tun sÄ±ralamasÄ± kontrolÃ¼
        if target_column:
            target_pos = df_processed.columns.get_loc(target_column) + 1
            print(f"Hedef sÃ¼tun pozisyonu: {target_pos}/{len(df_processed.columns)}")
        
        # BaÅŸarÄ± mesajlarÄ±
        if final_features <= initial_features + 10:
            print("âœ… Minimal yaklaÅŸÄ±m baÅŸarÄ±lÄ± - makul Ã¶zellik artÄ±ÅŸÄ±")
        else:
            print("âš ï¸ Hala fazla Ã¶zellik eklendi - kontrol edin")
        
        return df_processed