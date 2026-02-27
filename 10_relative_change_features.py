
import pandas as pd
import numpy as np
import os

base_path = r"D:\USveri"

input_files = [
    os.path.join(base_path, "jerk_output_0750_0805_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0805_0820_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0820_0835_with_preceding_enriched_exactonly.csv")
]

def _segment_from_path(path):
    return os.path.basename(path).replace(".csv", "")

output_files = [
    os.path.join(base_path, f"pct_logr_features_{_segment_from_path(p)}.csv") for p in input_files
]

merged_output = os.path.join(base_path, "pct_logr_features_ALLSEGMENTS.xlsx")

# Değişim hesaplama 
def compute_change(group, col, frame_offset, mode='pct'):
    group = group.sort_values('Frame_ID').reset_index(drop=True)
    frame_id_to_val = dict(zip(group['Frame_ID'], group[col]))
    values = []

    for idx, row in group.iterrows():
        base_frame = row['Frame_ID']
        base_value = row[col]

        target_frame = base_frame + frame_offset
        future_value = frame_id_to_val.get(target_frame, np.nan)

        if pd.notna(base_value) and pd.notna(future_value) and base_value != 0:
            if mode == 'pct':
                val = ((future_value - base_value) / base_value) * 100
            elif mode == 'logr':
                val = np.log(future_value / base_value) if base_value > 0 and future_value > 0 else np.nan
            else:
                val = np.nan
        else:
            val = np.nan

        values.append(val)

    return pd.Series(values, index=group.index)

# İstatistik hesaplama
def extract_stats(series, prefix, include_absm=True):
    stats = {}
    s = series.dropna()
    n = len(s)
    if include_absm:
        stats[f"{prefix}.absm"] = s.abs().mean() if n > 0 else np.nan
    stats[f"{prefix}.max"] = s.max() if n > 0 else np.nan
    stats[f"{prefix}.mean"] = s.mean() if n > 0 else np.nan
    stats[f"{prefix}.min"] = s.min() if n > 0 else np.nan
    stats[f"{prefix}.std"] = s.std() if n > 0 else np.nan
    stats[f"{prefix}.q2"] = s.quantile(0.5) if n > 0 else np.nan

    if n >= 5:
        stats[f"{prefix}.p01"] = np.percentile(s, 1)
        stats[f"{prefix}.p05"] = np.percentile(s, 5)
        stats[f"{prefix}.p95"] = np.percentile(s, 95)
        stats[f"{prefix}.p99"] = np.percentile(s, 99)
        stats[f"{prefix}.q1"] = s.quantile(0.25)
        stats[f"{prefix}.q3"] = s.quantile(0.75)
    else:
        for stat in ['p01', 'p05', 'p95', 'p99', 'q1', 'q3']:
            stats[f"{prefix}.{stat}"] = np.nan

    return stats

# Özellik çıkarım fonksiyonu (her zaman penceresi için ayrı)
def compute_all_features(df):
    features = []
    cols_pct = [
        'v_Vel_Smoothed', 'v_Acc_Smoothed', 'Gap',
        'vel_diff', 'acc_diff',
        'Preceding_v_Vel_Smoothed', 'Preceding_v_Acc_Smoothed',
        'Local_Y', 'Preceding_y'
    ]
    cols_logr = ['Local_Y', 'v_Vel_Smoothed', 'Gap', 'Preceding_y', 'Preceding_v_Vel_Smoothed']

    windows = {'w1': 10, 'w2': 50, 'w3': 100}

    for vid, group in df.groupby('Vehicle_ID'):
        row = {'Vehicle_ID': vid}
        for w_label, w_size in windows.items():
            for col in cols_pct:
                if col in group.columns:
                    pct_series = compute_change(group, col, frame_offset=w_size, mode='pct')
                    row.update(extract_stats(pct_series, f"{col}.pct_{w_label}", include_absm=True))

            for col in cols_logr:
                if col in group.columns:
                    logr_series = compute_change(group, col, frame_offset=w_size, mode='logr')
                    row.update(extract_stats(logr_series, f"{col}.logr_{w_label}", include_absm=False))
        features.append(row)
    return pd.DataFrame(features)

#Segmentleri işlenmesi
all_dfs = []
for in_path, out_path in zip(input_files, output_files):
    print(f"\ İşleniyor: {in_path}")
    df = pd.read_csv(in_path)
    df_feat = compute_all_features(df)
    # Araç ID'sine segment etiketi eklenmesi
    segment = _segment_from_path(in_path)
    df_feat["Vehicle_ID"] = df_feat["Vehicle_ID"].apply(lambda x: f"{segment}_{int(x)}")
    df_feat.to_csv(out_path, index=False)
    print(f"Kaydedildi: {out_path}")
    all_dfs.append(df_feat)
#Birleştir ve Excel olarak kaydet
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_excel(merged_output, index=False)
print(f"\Tüm segmentler tamamlandı. Birleşik çıktı (Excel): {merged_output}")
