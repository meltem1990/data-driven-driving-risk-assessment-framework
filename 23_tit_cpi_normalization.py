

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


file_path = r"D:\USveri\revizeÇALIŞMADOSYASIYENİMADR13_942_VehicleSummary.xlsx"
df = pd.read_excel(file_path)

#Normalizasyon 
cols_to_normalize = ["TIT_t1", "TIT_t2", "TIT_t3", "CPI_MADR1", "CPI_MADR2"]

# Min–Max normalizasyonu 
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

#  Yeni sütun adlarını '_norm' ekiyle eklenmesi
df_norm = df_norm.rename(columns={col: f"{col}_norm" for col in cols_to_normalize})

# Yeni dosya olarak kaydet
output_path = r"D:\USveri\makalerevision\30.11ÇalışmaDosyasıYeniMADR13_942_VehicleSummary_NORMALIZED.xlsx"
df_norm.to_excel(output_path, index=False)

print("min–Max normalizasyon tamamlandı.")
print(f" Normalleştirilmiş dosya kaydedildi: {output_path}")
