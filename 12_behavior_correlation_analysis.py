
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr

base_path = r"D:\USveri"
input_files = {
    "0750-0805": os.path.join(base_path, "merged_0750-0805.csv"),
    "0805-0820": os.path.join(base_path, "merged_0805-0820.csv"),
    "0820-0835": os.path.join(base_path, "merged_0820-0835.csv")
}
variables = [
    "v_Vel_Smoothed",
    "v_Acc_Smoothed",
    "Preceding_v_Vel_Smoothed",
    "Preceding_v_Acc_Smoothed"
]
target = "TTC"

windows = {
    "w1": 10,
    "w2": 50,
    "w3": 100
}
stats_to_compute = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "p05": lambda x: np.percentile(x, 5),
    "q1": lambda x: np.percentile(x, 25),
    "q2": lambda x: np.percentile(x, 50),
    "p95": lambda x: np.percentile(x, 95)
}

# Ana analiz fonksiyonu
def compute_correlations_for_vehicle(group, win_sizes):
    group = group.sort_values("Frame_ID")
    row_result = {"Vehicle_ID": group["Vehicle_ID"].iloc[0]}
    
    for w_label, w_size in win_sizes.items():
        for var in variables:
            pcor_list = []
            scor_list = []
            for i in range(len(group) - w_size + 1):
                sub = group.iloc[i:i + w_size]
                x = sub[var].values
                y = sub[target].values
                if np.isnan(x).any() or np.isnan(y).any():
                    continue
                if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                    continue  # sabit veri varsa korelasyon anlamsız
                try:
                    p = pearsonr(x, y)[0]
                    s = spearmanr(x, y)[0]
                    if np.isfinite(p):
                        pcor_list.append(p)
                    if np.isfinite(s):
                        scor_list.append(s)
                except Exception:
                    continue
            # istatistikleri hesapla
            for corr_type, values in [("pcor", pcor_list), ("scor", scor_list)]:
                if len(values) == 0:
                    # Boşsa tüm istatistikleri NaN yap
                    for stat_name in stats_to_compute:
                        colname = f"{var}-TTC-{w_label}_{corr_type}_{stat_name}"
                        row_result[colname] = np.nan
                else:
                    for stat_name, func in stats_to_compute.items():
                        colname = f"{var}-TTC-{w_label}_{corr_type}_{stat_name}"
                        row_result[colname] = func(values)
    return row_result
# Ana döngü
for seg, filepath in input_files.items():
    print(f"Segment işleniyor: {seg}")
    df = pd.read_csv(filepath)
    
    all_results = []
    for vid, group in df.groupby("Vehicle_ID"):
        res = compute_correlations_for_vehicle(group, windows)
        all_results.append(res)
    
    result_df = pd.DataFrame(all_results)
    
    # Kaydet
    out_path = os.path.join(base_path, f"correlation_summary_{seg}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Kaydedildi: {out_path}")
