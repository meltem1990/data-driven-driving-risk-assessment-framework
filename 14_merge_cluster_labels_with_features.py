

import pandas as pd
import os

base_path = r"D:\USveri"  

file_kume = os.path.join(base_path, "us_arac_kume_bilgileri_5_kume - Deneme.xlsx")
file_features = os.path.join(base_path, "reviseALL_FEATURES_MERGED.xlsx")

df_kume = pd.read_excel(file_kume)
df_feat = pd.read_excel(file_features)

df_merged = pd.merge(df_feat, df_kume, on="Vehicle_ID", how="left")

output_path = os.path.join(base_path, "reviseALL_FEATURES_FILTERED_WITH_KUME.xlsx")
df_merged.to_excel(output_path, index=False)

print(f"Tüm veriler küme bilgisiyle birleştirildi: {output_path}")
