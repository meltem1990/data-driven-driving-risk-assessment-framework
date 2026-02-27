
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r"D:\USveri\kendicalistiğimYENİMADR13_942_VehicleSummary - Kopya.xlsx"
df = pd.read_excel(file_path)

cols_to_normalize = ["TIT_t1", "TIT_t2", "TIT_t3", "CPI_MADR1", "CPI_MADR2"]

# Min–Max normalizasyonu uygulama
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

output_path = r"D:\USveri\kendicalistiğimYeniMADR13_942_VehicleSummary_NORMALIZED.xlsx"
df.to_excel(output_path, index=False)

print(" Min–Max normalizasyon tamamlandı.")
print(f"Normalleştirilmiş dosya kaydedildi: {output_path}")
