
import pandas as pd
import os
from functools import reduce


base_path = r"D:\USverifiles = [
    os.path.join(base_path, "vehicle_features_ALLSEGMENTS_updated.xlsx"),
    os.path.join(base_path, "relative_comparison_and_preceding_vehicle_all_features_ALLSEGMENTS_UPDATED.xlsx"),
    os.path.join(base_path, "2pct_logr_features_ALLSEGMENTS_MERGED.xlsx"),
    os.path.join(base_path, "microscale_features_w1_w2_w3_ALLSEGMENTS_UPDATED.xlsx"),
    os.path.join(base_path, "correlation_summary_all_segments.xlsx")
]

dfs = [pd.read_excel(f) for f in files]

# vehicle_ID üzerinden sırayla merge et
df_merged = reduce(lambda left, right: pd.merge(left, right, on="Vehicle_ID", how="outer"), dfs)

output_path = os.path.join(base_path, "xxxreviseALL_FEATURES_MERGED.xlsx")
df_merged.to_excel(output_path, index=False)

print(f"Tüm dosyalar birleştirildi ve kaydedildi: {output_path}")
