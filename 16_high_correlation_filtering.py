
import pandas as pd
import numpy as np
import os

base_path = r"D:\USveri\REVİZE29.10"
input_file = os.path.join(base_path, "ALL_FEATURES_MERGED.xlsx")
output_file = os.path.join(base_path, "denemeALL_FEATURES_REDUCTION_REPORT.xlsx")

df = pd.read_excel(input_file)
initial_cols = df.columns.tolist()
print(f" Başlangıç sütun sayısı: {len(initial_cols)}")

#Sabit veya düşük varyanslı sütunları bul 
numeric_cols = df.select_dtypes(include=[np.number]).columns
low_var_cols = [col for col in numeric_cols if df[col].std() == 0]
df = df.drop(columns=low_var_cols, errors="ignore")
print(f" Düşük varyanslı sütun sayısı: {len(low_var_cols)}")

#  %95’ten fazlası aynı değerde olan sütunları bul
high_const_cols = [
    col for col in df.columns 
    if (df[col].value_counts(normalize=True, dropna=False).iloc[0] > 0.95)
]
df = df.drop(columns=high_const_cols, errors="ignore")
print(f" %95'ten fazlası aynı değerde olan sütun sayısı: {len(high_const_cols)}")

#Yüksek korelasyonlu sütunları bul (>|0.85|) 
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for col in upper.columns:
    correlated = upper.index[upper[col] > 0.85].tolist()
    for c in correlated:
        high_corr_pairs.append((col, c, upper.loc[c, col]))

high_corr_cols = list(set([col for pair in high_corr_pairs for col in pair[:1]]))
df_reduced = df.drop(columns=high_corr_cols, errors="ignore")
print(f" Yüksek korelasyonlu sütun sayısı: {len(high_corr_cols)}")
print(f"Son kalan sütun sayısı: {df_reduced.shape[1]}")

# Excel raporu 
with pd.ExcelWriter(output_file) as writer:
    pd.DataFrame({"Removed_LowVar": low_var_cols}).to_excel(writer, sheet_name="LowVariance", index=False)
    pd.DataFrame({"Removed_HighConst": high_const_cols}).to_excel(writer, sheet_name="HighConstant", index=False)
    pd.DataFrame(high_corr_pairs, columns=["Feature_A", "Feature_B", "Corr_Value"]).to_excel(writer, sheet_name="HighCorrelation", index=False)
    pd.DataFrame({"Remaining_Features": df_reduced.columns}).to_excel(writer, sheet_name="Final_Remaining", index=False)

print(f"\ Ayrıntılı rapor kaydedildi: {output_file}")
