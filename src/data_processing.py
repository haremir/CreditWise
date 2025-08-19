import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CreditDataProcessor:
    """
    Kredi skoru verisi için kapsamlı veri ön işleme sınıfı
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
        
        self.target_column = 'Credit_Score'  # Hedef değişken
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Veriyi yükle
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Veri başarıyla yüklendi. Boyut: {df.shape}")
            return df
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None
    
    def basic_info(self, df: pd.DataFrame) -> None:
        """
        Temel veri bilgileri
        """
        print("=== VERİ BİLGİLERİ ===")
        print(f"Boyut: {df.shape}")
        print(f"Sütunlar: {list(df.columns)}")
        print(f"\nVeri tipleri:")
        print(df.dtypes)
        print(f"\nEksik değer sayıları:")
        print(df.isnull().sum().sort_values(ascending=False))
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik değerleri işle
        """
        df_processed = df.copy()
        
        print("=== EKSİK DEĞER İŞLEME ===")
        
        # '_' ve boş string değerlerini NaN yap
        df_processed = df_processed.replace(['_', '', ' ', 'nan', 'NaN', 'null', 'NULL'], np.nan)
        
        # Sayısal sütunlarda eksik değerleri medyan ile doldur
        for col in self.numeric_columns:
            if col in df_processed.columns:
                # Önce sayısal olmayan değerleri temizle ve sayıya çevir
                original_nulls = df_processed[col].isnull().sum()
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                conversion_nulls = df_processed[col].isnull().sum() - original_nulls
                
                if conversion_nulls > 0:
                    print(f"{col}: {conversion_nulls} geçersiz değer NaN'a çevrildi")
                
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    if pd.isna(median_val):  # Eğer medyan da NaN ise, 0 kullan
                        median_val = 0
                    df_processed[col].fillna(median_val, inplace=True)
                    print(f"{col}: {df_processed[col].isnull().sum()} eksik değer medyan ({median_val:.2f}) ile dolduruldu")
        
        # Kategorik sütunlarda eksik değerleri mod ile doldur
        for col in self.categorical_columns:
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    mode_vals = df_processed[col].mode()
                    mode_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'Unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                    print(f"{col}: eksik değerler '{mode_val}' ile dolduruldu")
        
        # Credit_History_Age özel işlem
        if 'Credit_History_Age' in df_processed.columns:
            df_processed['Credit_History_Age'].fillna('0 Years and 0 Months', inplace=True)
            print("Credit_History_Age: eksik değerler '0 Years and 0 Months' ile dolduruldu")
        
        # Son kontrol: kalan NaN değerleri göster
        remaining_nulls = df_processed.isnull().sum()
        if remaining_nulls.sum() > 0:
            print("\nKalan eksik değerler:")
            print(remaining_nulls[remaining_nulls > 0])
        else:
            print("✅ Tüm eksik değerler işlendi")
        
        return df_processed
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sayısal sütunları temizle
        """
        df_cleaned = df.copy()
        
        print("=== SAYISAL SÜTUN TEMİZLEME ===")
        
        # Önce tüm sayısal sütunları gerçekten sayısal formata çevir
        for col in self.numeric_columns:
            if col in df_cleaned.columns:
                # String değerleri sayıya çevir, çevrilemeyenleri NaN yap
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Age sütunu: Negatif değerleri pozitif yap
        if 'Age' in df_cleaned.columns:
            # Önce NaN değerleri kontrol et ve doldur
            if df_cleaned['Age'].isnull().sum() > 0:
                median_age = df_cleaned['Age'].median()
                df_cleaned['Age'].fillna(median_age, inplace=True)
                print(f"Age: {df_cleaned['Age'].isnull().sum()} NaN değer medyan ile dolduruldu")
            
            negative_ages = (df_cleaned['Age'] < 0).sum()
            df_cleaned['Age'] = df_cleaned['Age'].abs()
            if negative_ages > 0:
                print(f"Age: {negative_ages} negatif değer pozitif yapıldı")
        
        # Faiz oranı sütunu: Aykırı değerleri sınırla
        if 'Interest_Rate' in df_cleaned.columns:
            # NaN değerleri kontrol et
            if df_cleaned['Interest_Rate'].isnull().sum() > 0:
                median_rate = df_cleaned['Interest_Rate'].median()
                df_cleaned['Interest_Rate'].fillna(median_rate, inplace=True)
                print(f"Interest_Rate: NaN değerler medyan ile dolduruldu")
            
            q75 = df_cleaned['Interest_Rate'].quantile(0.75)
            q25 = df_cleaned['Interest_Rate'].quantile(0.25)
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr
            outliers = (df_cleaned['Interest_Rate'] > upper_bound).sum()
            df_cleaned['Interest_Rate'] = np.where(
                df_cleaned['Interest_Rate'] > upper_bound,
                upper_bound,
                df_cleaned['Interest_Rate']
            )
            if outliers > 0:
                print(f"Interest_Rate: {outliers} aykırı değer sınırlandı")
        
        # Credit_Utilization_Ratio: 0-100 arasında sınırla
        if 'Credit_Utilization_Ratio' in df_cleaned.columns:
            # NaN değerleri kontrol et
            if df_cleaned['Credit_Utilization_Ratio'].isnull().sum() > 0:
                median_ratio = df_cleaned['Credit_Utilization_Ratio'].median()
                df_cleaned['Credit_Utilization_Ratio'].fillna(median_ratio, inplace=True)
                print(f"Credit_Utilization_Ratio: NaN değerler medyan ile dolduruldu")
            
            over_100 = (df_cleaned['Credit_Utilization_Ratio'] > 100).sum()
            under_0 = (df_cleaned['Credit_Utilization_Ratio'] < 0).sum()
            df_cleaned['Credit_Utilization_Ratio'] = np.clip(
                df_cleaned['Credit_Utilization_Ratio'], 0, 100
            )
            if over_100 > 0 or under_0 > 0:
                print(f"Credit_Utilization_Ratio: {over_100 + under_0} değer 0-100 arasında sınırlandı")
        
        # Diğer sayısal sütunlarda kalan NaN değerleri kontrol et ve doldur
        for col in self.numeric_columns:
            if col in df_cleaned.columns and col not in ['Age', 'Interest_Rate', 'Credit_Utilization_Ratio']:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    print(f"{col}: NaN değerler medyan ({median_val:.2f}) ile dolduruldu")
        
        return df_cleaned
    
    def process_credit_history_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Credit_History_Age sütununu sayısal değere dönüştür
        """
        df_processed = df.copy()
        
        if 'Credit_History_Age' in df_processed.columns:
            def parse_credit_age(age_str):
                if pd.isna(age_str) or age_str == 'NA':
                    return 0
                
                try:
                    years = 0
                    months = 0
                    
                    # Years bilgisini çıkar
                    if 'Years' in str(age_str):
                        year_part = str(age_str).split('Years')[0].strip()
                        if year_part.replace('-', '').isdigit():
                            years = abs(int(year_part))
                    
                    # Months bilgisini çıkar
                    if 'Months' in str(age_str):
                        month_part = str(age_str).split('and')[-1].split('Months')[0].strip()
                        if month_part.replace('-', '').isdigit():
                            months = abs(int(month_part))
                    
                    return years * 12 + months
                except:
                    return 0
            
            df_processed['Credit_History_Age_Months'] = df_processed['Credit_History_Age'].apply(parse_credit_age)
            print("Credit_History_Age ay cinsinden sayısal değere dönüştürüldü")
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kategorik değişkenleri kodla
        """
        df_encoded = df.copy()
        
        print("=== KATEGORİK KODLAMA ===")
        
        # Month sütunu: Sayısal sıraya göre kodla
        if 'Month' in df_encoded.columns:
            month_mapping = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df_encoded['Month_Numeric'] = df_encoded['Month'].map(month_mapping)
            print("Month sütunu sayısal değerlere dönüştürüldü")
        
        # Payment_of_Min_Amount: Binary kodlama
        if 'Payment_of_Min_Amount' in df_encoded.columns:
            df_encoded['Payment_of_Min_Amount_Binary'] = df_encoded['Payment_of_Min_Amount'].map(
                {'Yes': 1, 'No': 0}
            )
            print("Payment_of_Min_Amount binary kodlandı")
        
        # Credit_Mix: Ordinal kodlama
        if 'Credit_Mix' in df_encoded.columns:
            credit_mix_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2}
            df_encoded['Credit_Mix_Encoded'] = df_encoded['Credit_Mix'].map(credit_mix_mapping)
            print("Credit_Mix ordinal kodlandı")
        
        # Payment_Behaviour: One-hot encoding için hazırla
        if 'Payment_Behaviour' in df_encoded.columns:
            payment_dummies = pd.get_dummies(df_encoded['Payment_Behaviour'], prefix='Payment_Behaviour')
            df_encoded = pd.concat([df_encoded, payment_dummies], axis=1)
            print("Payment_Behaviour one-hot kodlandı")
        
        # Occupation: One-hot encoding
        if 'Occupation' in df_encoded.columns:
            occupation_dummies = pd.get_dummies(df_encoded['Occupation'], prefix='Occupation')
            df_encoded = pd.concat([df_encoded, occupation_dummies], axis=1)
            print("Occupation one-hot kodlandı")
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yeni özellikler oluştur - INFINITY kontrolü eklendi
        """
        df_featured = df.copy()
        
        print("=== YENİ ÖZELLİK OLUŞTURMA ===")
        
        # Gelir-Gider oranı
        if 'Annual_Income' in df_featured.columns and 'Total_EMI_per_month' in df_featured.columns:
            df_featured['Monthly_Income'] = df_featured['Annual_Income'] / 12
            # Sıfıra bölmeyi önle
            df_featured['Income_to_EMI_Ratio'] = df_featured['Monthly_Income'] / (df_featured['Total_EMI_per_month'] + 1)
            print("Gelir-EMI oranı oluşturuldu")
        
        # Yatırım oranı
        if 'Amount_invested_monthly' in df_featured.columns and 'Monthly_Inhand_Salary' in df_featured.columns:
            # Sıfıra bölmeyi önle
            df_featured['Investment_Ratio'] = df_featured['Amount_invested_monthly'] / (df_featured['Monthly_Inhand_Salary'] + 1)
            print("Yatırım oranı oluşturuldu")
        
        # Kredi kartı başına ortalama borç
        if 'Outstanding_Debt' in df_featured.columns and 'Num_Credit_Card' in df_featured.columns:
            # Sıfıra bölmeyi önle
            df_featured['Debt_per_Card'] = df_featured['Outstanding_Debt'] / (df_featured['Num_Credit_Card'] + 1)
            print("Kart başına borç oluşturuldu")
        
        # Gecikme skoru
        if 'Num_of_Delayed_Payment' in df_featured.columns and 'Delay_from_due_date' in df_featured.columns:
            df_featured['Delay_Score'] = df_featured['Num_of_Delayed_Payment'] * df_featured['Delay_from_due_date']
            print("Gecikme skoru oluşturuldu")
        
        # Yaş grubu
        if 'Age' in df_featured.columns:
            df_featured['Age_Group'] = pd.cut(
                df_featured['Age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elder']
            )
            age_group_dummies = pd.get_dummies(df_featured['Age_Group'], prefix='Age_Group')
            df_featured = pd.concat([df_featured, age_group_dummies], axis=1)
            print("Yaş grupları oluşturuldu")
        
        # Infinity değerlerini kontrol et ve temizle
        df_featured = self.check_and_fix_infinity(df_featured)
        
        return df_featured
    
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
                median_val = df_fixed[col][~np.isinf(df_fixed[col])].median()
                if pd.isna(median_val):
                    median_val = 0
                
                df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], median_val)
                print(f"  ✅ {col}: infinity değerler {median_val:.2f} ile değiştirildi")
        
        if not infinity_found:
            print("✅ Infinity değer bulunamadı")
        
        # NaN değerleri de kontrol et
        nan_cols = df_fixed.isnull().sum()
        if nan_cols.sum() > 0:
            print("\n⚠️ Kalan NaN değerler:")
            for col, count in nan_cols[nan_cols > 0].items():
                if col in numeric_cols:
                    median_val = df_fixed[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_fixed[col].fillna(median_val, inplace=True)
                    print(f"  ✅ {col}: {count} NaN değer {median_val:.2f} ile dolduruldu")
        
        return df_fixed
    
    def remove_outliers(self, df: pd.DataFrame, method='iqr', threshold=1.5) -> pd.DataFrame:
        """
        Aykırı değerleri kaldır
        """
        df_clean = df.copy()
        initial_shape = df_clean.shape[0]
        
        print(f"=== AYKIRI DEĞER TEMİZLEME ({method.upper()}) ===")
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                if col in self.numeric_columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    ]
                    
                    if outliers_before > 0:
                        print(f"{col}: {outliers_before} aykırı değer kaldırıldı")
        
        final_shape = df_clean.shape[0]
        print(f"Toplam {initial_shape - final_shape} satır kaldırıldı")
        
        return df_clean
    
    def process_pipeline(self, df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
        """
        Tam veri işleme pipeline'ı
        """
        print("=== VERİ İŞLEME PIPELINE BAŞLADI ===\n")
        
        # 1. Temel bilgiler
        self.basic_info(df)
        
        # 2. Eksik değer işleme
        df = self.handle_missing_values(df)
        
        # 3. Sayısal sütun temizleme
        df = self.clean_numeric_columns(df)
        
        # 4. Credit history age işleme
        df = self.process_credit_history_age(df)
        
        # 5. Kategorik kodlama
        df = self.encode_categorical_variables(df)
        
        # 6. Yeni özellik oluşturma (infinity kontrolü ile)
        df = self.create_features(df)
        
        # 7. Aykırı değer temizleme (isteğe bağlı)
        if remove_outliers:
            df = self.remove_outliers(df)
        
        # 8. Final infinity ve NaN kontrolü
        df = self.check_and_fix_infinity(df)
        
        print("\n=== VERİ İŞLEME TAMAMLANDI ===")
        print(f"Final boyut: {df.shape}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str) -> None:
        """
        İşlenmiş veriyi kaydet
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"İşlenmiş veri kaydedildi: {file_path}")
        except Exception as e:
            print(f"Kaydetme hatası: {e}")