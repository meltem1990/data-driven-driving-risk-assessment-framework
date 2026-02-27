
import pandas as pd
from scipy.stats import spearmanr

file_path = r"D:\USveri\30.11ÇalışmaDosyasıYeniMADR13_942_VehicleSummary_NORMALIZED.xlsx"
df = pd.read_excel(file_path)
# göstergeleri ve etiket sütununu tanımla
cols = ["TIT_t1_norm", "TIT_t2_norm", "TIT_t3_norm", "CPI_MADR1_norm", "CPI_MADR2_norm"]
label_col = "Risk_Label_5"
# Spearman rho katsayılarını hesapla
rho_values = {}
for col in cols:
    rho, _ = spearmanr(df[col], df[label_col])
    rho_values[col] = rho
# Ağırlıkları hesapla (rho / toplam_rho)
total_rho = sum(rho_values.values())
weights = {col: rho_values[col] / total_rho for col in cols}

# Her satır için risk skorunu hesapla 
df["Risk_Score"] = sum(df[col] * weights[col] for col in cols)

# Risk_Score için Min–Max normalizasyonu 
min_score = df["Risk_Score"].min()
max_score = df["Risk_Score"].max()

df["Risk_Score_norm"] = (df["Risk_Score"] - min_score) / (max_score - min_score)

# Rho ve ağırlıkları ayrı tabloya kaydet 
rho_weight_df = pd.DataFrame({
    "Gösterge": cols,
    "Spearman_rho": [rho_values[c] for c in cols],
    "Ağırlık": [weights[c] for c in cols]
})
#Excel'e kaydet 
output_path = r"D:\USveri\makalerevision\30.11ÇalışmaDosyasıYeniMADR13_942_VehicleSummary_RISK_SCORE.xlsx"
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, index=False, sheet_name="Risk_Score_Data")
    rho_weight_df.to_excel(writer, index=False, sheet_name="Spearman_Weights")

print(" Spearman rho, ağırlıklar ve risk skorları başarıyla hesaplandı.")
print(f"Sonuç dosyası: {output_path}")
