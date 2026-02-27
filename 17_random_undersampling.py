

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import pandas as pd, numpy as np

file_path = "d:/USveri/yeniALL_FEATURES_FILTERED_WITH_KUME.xlsx"
target_col = 'Risk_Label_5'
df = pd.read_excel(file_path)

# Hedef ve özellikleri ayır
X = df.drop(columns=[target_col, 'Vehicle_ID'], errors='ignore')
y = df[target_col]

# Label encoding (gerekirse)
if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)
    df[target_col] = y  

# Eksik değerleri işle
X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how='all')
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

df[target_col] = y  
df['Vehicle_ID'] = df.get('Vehicle_ID', pd.Series(np.arange(len(df))))  

# R0 sınıfı (0) olanları ayır ve azalt
df_safe = df[df[target_col] == 0]
df_other = df[df[target_col] != 0]
df_safe_downsampled = resample(df_safe, replace=False, n_samples=1000, random_state=42)

# Birleştir
df_resampled = pd.concat([df_safe_downsampled, df_other])
df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Kaydet
output_path = "r2.11SAFE_1000_OTHERS_UNTOUCHED.xlsx"
df_resampled.to_excel(output_path, index=False)

#Bilgi çıktısı
print("R0 sınıf 1000 örneğe indirildi. Diğer sınıflar korundu.")
print("Yeni sınıf dağılımı:\n", df_resampled[target_col].value_counts())
print(f" Kaydedilen dosya: {output_path}")
