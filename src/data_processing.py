import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CreditDataProcessor:
    """
    Kredi skoru verisi iÃ§in temiz veri Ã¶n iÅŸleme sÄ±nÄ±fÄ±
    Minimal yaklaÅŸÄ±m - sadece temel temizlik ve kodlama
    """
    
    def __init__(self):
        self.numeric_columns = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
            'Amount_invested_monthly', 'Monthly_Balance'
        ]
        
        self.categorical_columns = [
            'Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 
            'Payment_of_Min_Amount', 'Payment_Behaviour'
        ]
        
        self.target_column = 'Credit_Score'
        
        # SaklanmasÄ± gereken orijinal sÃ¼tunlar
        self.keep_original_columns = [
            'ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation',
            'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age', 
            'Payment_of_Min_Amount', 'Payment_Behaviour'
        ]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Veriyi yÃ¼kle
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Veri baÅŸarÄ±yla yÃ¼klendi. Boyut: {df.shape}")
            return df
        except Exception as e:
            print(f"Veri yÃ¼kleme hatasÄ±: {e}")
            return None
    
    def basic_info(self, df: pd.DataFrame) -> None:
        """
        Temel veri bilgileri
        """
        print("=== VERÄ° BÄ°LGÄ°LERÄ° ===")
        print(f"Boyut: {df.shape}")
        print(f"SÃ¼tun sayÄ±sÄ±: {len(df.columns)}")
        print(f"\nVeri tipleri:")
        print(df.dtypes.value_counts())
        print(f"\nEksik deÄŸer sayÄ±larÄ±:")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("Eksik deÄŸer bulunamadÄ±")
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik deÄŸerleri iÅŸle
        """
        df_processed = df.copy()
        
        print("=== EKSÄ°K DEÄžER Ä°ÅžLEME ===")
        
        # '_' ve boÅŸ string deÄŸerlerini NaN yap
        df_processed = df_processed.replace(['_', '', ' ', 'nan', 'NaN', 'null', 'NULL'], np.nan)
        
        # SayÄ±sal sÃ¼tunlarda eksik deÄŸerleri medyan ile doldur
        for col in self.numeric_columns:
            if col in df_processed.columns:
                # Ã–nce sayÄ±sal olmayan deÄŸerleri temizle ve sayÄ±ya Ã§evir
                original_nulls = df_processed[col].isnull().sum()
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                conversion_nulls = df_processed[col].isnull().sum() - original_nulls
                
                if conversion_nulls > 0:
                    print(f"{col}: {conversion_nulls} geÃ§ersiz deÄŸer NaN'a Ã§evrildi")
                
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    if pd.isna(median_val):  # EÄŸer medyan da NaN ise, 0 kullan
                        median_val = 0
                    df_processed[col].fillna(median_val, inplace=True)
                    print(f"{col}: {df_processed[col].isnull().sum()} eksik deÄŸer medyan ({median_val:.2f}) ile dolduruldu")
        
        # Kategorik sÃ¼tunlarda eksik deÄŸerleri mod ile doldur
        for col in self.categorical_columns:
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    mode_vals = df_processed[col].mode()
                    mode_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'Unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                    print(f"{col}: eksik deÄŸerler '{mode_val}' ile dolduruldu")
        
        # Credit_History_Age Ã¶zel iÅŸlem
        if 'Credit_History_Age' in df_processed.columns:
            df_processed['Credit_History_Age'].fillna('0 Years and 0 Months', inplace=True)
            print("Credit_History_Age: eksik deÄŸerler '0 Years and 0 Months' ile dolduruldu")
        
        # Son kontrol
        remaining_nulls = df_processed.isnull().sum()
        if remaining_nulls.sum() > 0:
            print("\nKalan eksik deÄŸerler:")
            print(remaining_nulls[remaining_nulls > 0])
        else:
            print("TÃ¼m eksik deÄŸerler iÅŸlendi")
        
        return df_processed
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SayÄ±sal sÃ¼tunlarÄ± temizle
        """
        df_cleaned = df.copy()
        
        print("=== SAYISAL SÃœTUN TEMÄ°ZLEME ===")
        
        # Ã–nce tÃ¼m sayÄ±sal sÃ¼tunlarÄ± gerÃ§ekten sayÄ±sal formata Ã§evir
        for col in self.numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Age sÃ¼tunu: Negatif deÄŸerleri pozitif yap ve mantÄ±klÄ± aralÄ±k
        if 'Age' in df_cleaned.columns:
            if df_cleaned['Age'].isnull().sum() > 0:
                median_age = df_cleaned['Age'].median()
                df_cleaned['Age'].fillna(median_age, inplace=True)
                print(f"Age: NaN deÄŸerler medyan ile dolduruldu")
            
            negative_ages = (df_cleaned['Age'] < 0).sum()
            df_cleaned['Age'] = df_cleaned['Age'].abs()
            df_cleaned['Age'] = np.clip(df_cleaned['Age'], 18, 100)
            if negative_ages > 0:
                print(f"Age: {negative_ages} negatif deÄŸer pozitif yapÄ±ldÄ± ve 18-100 arasÄ± sÄ±nÄ±rlandÄ±")
        
        # Faiz oranÄ± sÃ¼tunu: AykÄ±rÄ± deÄŸerleri sÄ±nÄ±rla
        if 'Interest_Rate' in df_cleaned.columns:
            if df_cleaned['Interest_Rate'].isnull().sum() > 0:
                median_rate = df_cleaned['Interest_Rate'].median()
                df_cleaned['Interest_Rate'].fillna(median_rate, inplace=True)
                print(f"Interest_Rate: NaN deÄŸerler medyan ile dolduruldu")
            
            out_of_range = ((df_cleaned['Interest_Rate'] < 0) | (df_cleaned['Interest_Rate'] > 50)).sum()
            df_cleaned['Interest_Rate'] = np.clip(df_cleaned['Interest_Rate'], 0, 50)
            if out_of_range > 0:
                print(f"Interest_Rate: {out_of_range} deÄŸer 0-50 aralÄ±ÄŸÄ±nda sÄ±nÄ±rlandÄ±")
        
        # Credit_Utilization_Ratio: 0-100 arasÄ± sÄ±nÄ±rla
        if 'Credit_Utilization_Ratio' in df_cleaned.columns:
            if df_cleaned['Credit_Utilization_Ratio'].isnull().sum() > 0:
                median_ratio = df_cleaned['Credit_Utilization_Ratio'].median()
                df_cleaned['Credit_Utilization_Ratio'].fillna(median_ratio, inplace=True)
                print(f"Credit_Utilization_Ratio: NaN deÄŸerler medyan ile dolduruldu")
            
            over_100 = (df_cleaned['Credit_Utilization_Ratio'] > 100).sum()
            under_0 = (df_cleaned['Credit_Utilization_Ratio'] < 0).sum()
            df_cleaned['Credit_Utilization_Ratio'] = np.clip(
                df_cleaned['Credit_Utilization_Ratio'], 0, 100
            )
            if over_100 > 0 or under_0 > 0:
                print(f"Credit_Utilization_Ratio: {over_100 + under_0} deÄŸer 0-100 arasÄ± sÄ±nÄ±rlandÄ±")
        
        # DiÄŸer sayÄ±sal sÃ¼tunlarda kalan NaN deÄŸerleri kontrol et ve doldur
        for col in self.numeric_columns:
            if col in df_cleaned.columns and col not in ['Age', 'Interest_Rate', 'Credit_Utilization_Ratio']:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_cleaned[col].fillna(median_val, inplace=True)
                    print(f"{col}: NaN deÄŸerler medyan ({median_val:.2f}) ile dolduruldu")
        
        return df_cleaned
    
    def process_credit_history_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Credit_History_Age sÃ¼tununu sayÄ±sal deÄŸere dÃ¶nÃ¼ÅŸtÃ¼r
        """
        df_processed = df.copy()
        
        if 'Credit_History_Age' in df_processed.columns:
            def parse_credit_age(age_str):
                if pd.isna(age_str) or age_str == 'NA':
                    return 0
                
                try:
                    years = 0
                    months = 0
                    
                    # Years bilgisini Ã§Ä±kar
                    if 'Years' in str(age_str):
                        year_part = str(age_str).split('Years')[0].strip()
                        if year_part.replace('-', '').isdigit():
                            years = abs(int(year_part))
                    
                    # Months bilgisini Ã§Ä±kar
                    if 'Months' in str(age_str):
                        month_part = str(age_str).split('and')[-1].split('Months')[0].strip()
                        if month_part.replace('-', '').isdigit():
                            months = abs(int(month_part))
                    
                    return years * 12 + months
                except:
                    return 0
            
            df_processed['Credit_History_Age_Months'] = df_processed['Credit_History_Age'].apply(parse_credit_age)
            print("Credit_History_Age ay cinsinden sayÄ±sal deÄŸere dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼")
        
        return df_processed
    
    def encode_categorical_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SADECE TEMEL kategorik kodlama - frequency encoding yok
        """
        df_encoded = df.copy()
        
        print("=== TEMEL KATEGORÄ°K KODLAMA ===")
        
        # Month: SayÄ±sal sÄ±raya gÃ¶re
        if 'Month' in df_encoded.columns:
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df_encoded['Month_Numeric'] = df_encoded['Month'].map(month_mapping)
            df_encoded['Month_Numeric'].fillna(6, inplace=True)  # June
            print("Month sayÄ±sal kodlandÄ±")
        
        # Payment_of_Min_Amount: Binary
        if 'Payment_of_Min_Amount' in df_encoded.columns:
            df_encoded['Payment_Min_Binary'] = (df_encoded['Payment_of_Min_Amount'] == 'Yes').astype(int)
            print("Payment_of_Min_Amount binary kodlandÄ±")
        
        # Credit_Mix: Ordinal (anlamlÄ± sÄ±ralama)
        if 'Credit_Mix' in df_encoded.columns:
            credit_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2}
            df_encoded['Credit_Mix_Numeric'] = df_encoded['Credit_Mix'].map(credit_mapping)
            df_encoded['Credit_Mix_Numeric'].fillna(1, inplace=True)  # Standard
            print("Credit_Mix ordinal kodlandÄ±")
        
        # Payment_Behaviour: Label encoding (frequency deÄŸil!)
        if 'Payment_Behaviour' in df_encoded.columns:
            unique_behaviors = sorted(df_encoded['Payment_Behaviour'].dropna().unique())
            behavior_mapping = {behavior: idx for idx, behavior in enumerate(unique_behaviors)}
            df_encoded['Payment_Behaviour_Encoded'] = df_encoded['Payment_Behaviour'].map(behavior_mapping)
            df_encoded['Payment_Behaviour_Encoded'].fillna(0, inplace=True)
            print("Payment_Behaviour label kodlandÄ±")
        
        # Occupation: Label encoding (frequency deÄŸil!)
        if 'Occupation' in df_encoded.columns:
            unique_occupations = sorted(df_encoded['Occupation'].dropna().unique())
            occupation_mapping = {occ: idx for idx, occ in enumerate(unique_occupations)}
            df_encoded['Occupation_Encoded'] = df_encoded['Occupation'].map(occupation_mapping)
            df_encoded['Occupation_Encoded'].fillna(0, inplace=True)
            print("Occupation label kodlandÄ±")
        
        return df_encoded
    
    def minimal_feature_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SADECE 2-3 TEMEL tÃ¼retilmiÅŸ Ã¶zellik
        """
        df_featured = df.copy()
        
        print("=== MÄ°NÄ°MAL Ã–ZELLÄ°K OLUÅžTURMA ===")
        
        feature_count = 0
        
        # 1. AylÄ±k gelir (temel gereksinim)
        if 'Annual_Income' in df_featured.columns:
            df_featured['Monthly_Income'] = df_featured['Annual_Income'] / 12
            feature_count += 1
            print("Monthly_Income oluÅŸturuldu")
        
        # 2. Gelir-EMI oranÄ± (en kritik finansal gÃ¶sterge)
        if all(col in df_featured.columns for col in ['Monthly_Income', 'Total_EMI_per_month']):
            df_featured['Income_EMI_Ratio'] = df_featured['Monthly_Income'] / (df_featured['Total_EMI_per_month'] + 1)
            feature_count += 1
            print("Income_EMI_Ratio oluÅŸturuldu")
        
        # 3. Toplam gecikme etkisi (basit risk gÃ¶stergesi)
        if all(col in df_featured.columns for col in ['Num_of_Delayed_Payment', 'Delay_from_due_date']):
            delay_days = np.maximum(0, df_featured['Delay_from_due_date'])
            df_featured['Total_Delay_Score'] = df_featured['Num_of_Delayed_Payment'] * delay_days
            feature_count += 1
            print("Total_Delay_Score oluÅŸturuldu")
        
        # Infinity kontrolÃ¼
        df_featured = self.check_and_fix_infinity(df_featured)
        
        print(f"Toplam {feature_count} minimal Ã¶zellik oluÅŸturuldu")
        
        return df_featured
    
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
                
                # Infinity deÄŸerleri medyan ile deÄŸiÅŸtir
                finite_values = df_fixed[col][np.isfinite(df_fixed[col])]
                if len(finite_values) > 0:
                    median_val = finite_values.median()
                else:
                    median_val = 0
                
                df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], median_val)
                print(f"    {col}: infinity deÄŸerler {median_val:.2f} ile deÄŸiÅŸtirildi")
        
        if not infinity_found:
            print("Infinity deÄŸer bulunamadÄ±")
        
        # NaN deÄŸerleri de kontrol et
        nan_cols = df_fixed.isnull().sum()
        if nan_cols.sum() > 0:
            print("Kalan NaN deÄŸerler:")
            for col, count in nan_cols[nan_cols > 0].items():
                if col in numeric_cols:
                    median_val = df_fixed[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_fixed[col].fillna(median_val, inplace=True)
                    print(f"    {col}: {count} NaN deÄŸer {median_val:.2f} ile dolduruldu")
        
        return df_fixed
    
    def clean_processing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SADECE TEMEL VERÄ° TEMÄ°ZLEME - feature engineering yok
        """
        print("=== TEMÄ°Z VERÄ° Ä°ÅžLEME PIPELINE ===\n")
        
        df_processed = df.copy()
        
        # 1. Temel bilgiler
        self.basic_info(df_processed)
        
        # 2. Eksik deÄŸer iÅŸleme
        df_processed = self.handle_missing_values(df_processed)
        
        # 3. SayÄ±sal sÃ¼tun temizleme
        df_processed = self.clean_numeric_columns(df_processed)
        
        # 4. Credit History Age iÅŸleme
        df_processed = self.process_credit_history_age(df_processed)
        
        # 5. SADECE TEMEL kategorik kodlama
        df_processed = self.encode_categorical_basic(df_processed)
        
        # 6. Minimal Ã¶zellik oluÅŸturma (sadece 2-3 tane)
        df_processed = self.minimal_feature_creation(df_processed)
        
        # 7. Final temizlik
        df_processed = self.check_and_fix_infinity(df_processed)
        
        print(f"\n=== VERÄ° TEMÄ°ZLEME TAMAMLANDI ===")
        print(f"Final boyut: {df_processed.shape}")
        print(f"Toplam Ã¶zellik sayÄ±sÄ±: {df_processed.shape[1]}")
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Ä°ÅŸlenmiÅŸ veriyi kaydet
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Ä°ÅŸlenmiÅŸ veri kaydedildi: {file_path}")
        except Exception as e:
            print(f"Kaydetme hatasÄ±: {e}")
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Ä°ÅŸlenmiÅŸ verinin Ã¶zellik Ã¶zetini ver
        """
        # Orijinal sÃ¼tunlar
        original_features = [col for col in self.keep_original_columns + self.numeric_columns 
                           if col in df.columns]
        
        # TÃ¼retilmiÅŸ sÃ¼tunlar
        engineered_features = [col for col in df.columns 
                             if col not in original_features]
        
        summary = {
            'total_features': df.shape[1],
            'original_features': len(original_features),
            'engineered_features': len(engineered_features),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'original_feature_list': original_features,
            'engineered_feature_list': engineered_features
        }
        
        print("=== Ã–ZELLÄ°K Ã–ZETÄ° ===")
        print(f"Toplam Ã¶zellik: {summary['total_features']}")
        print(f"Orijinal Ã¶zellik: {summary['original_features']}")
        print(f"TÃ¼retilmiÅŸ Ã¶zellik: {summary['engineered_features']}")
        print(f"SayÄ±sal Ã¶zellik: {summary['numeric_features']}")
        print(f"Kategorik Ã¶zellik: {summary['categorical_features']}")
        
        if summary['engineered_features'] > 0:
            print(f"\nTÃ¼retilmiÅŸ Ã¶zellikler: {engineered_features}")
        
        return summary