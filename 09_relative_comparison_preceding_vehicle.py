
import os
import pandas as pd
import numpy as np

base_path = r"D:\USveri"

input_files = [
    os.path.join(base_path, "jerk_output_0750_0805_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0805_0820_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0820_0835_with_preceding_enriched_exactonly.csv"),
]

def _segment_from_path(p):
    return os.path.basename(p).replace(".csv", "")

# Çıkış dosyaları 
output_files = [
    os.path.join(base_path, f"vehicle_all_features_{_segment_from_path(p)}.csv") for p in input_files
]

# Birleşik çıktı dosyası
output_merged = os.path.join(base_path, "relative_comparison_and_preceding_vehicle_all_features_ALLSEGMENTS.csv")

# Kullanılacak istatistik listeleri
stats_diff = ['max', 'mean', 'min', 'p01', 'p05', 'p95', 'p99', 'q1', 'q2', 'q3', 'std']
stats_full = ['kurt', 'mad', 'max', 'mean', 'min', 'p01', 'p05', 'p95', 'p99', 'q1', 'q2', 'q3', 'skew', 'std']

#  İstatistik hesaplama fonksiyonu
def calculate_features(df, col, stats):
    feature_dict = {}
    if col not in df.columns:
        for stat in stats:
            feature_dict[f"{col}.{stat}"] = np.nan
        return feature_dict

    series = df[col].dropna()
    for stat in stats:
        try:
            if len(series) == 0:
                feature_dict[f"{col}.{stat}"] = np.nan
            elif stat == 'p01':
                feature_dict[f"{col}.p01"] = np.percentile(series, 1)
            elif stat == 'p05':
                feature_dict[f"{col}.p05"] = np.percentile(series, 5)
            elif stat == 'p95':
                feature_dict[f"{col}.p95"] = np.percentile(series, 95)
            elif stat == 'p99':
                feature_dict[f"{col}.p99"] = np.percentile(series, 99)
            elif stat == 'q1':
                feature_dict[f"{col}.q1"] = np.percentile(series, 25)
            elif stat == 'q2':
                feature_dict[f"{col}.q2"] = np.percentile(series, 50)
            elif stat == 'q3':
                feature_dict[f"{col}.q3"] = np.percentile(series, 75)
            elif stat == 'mean':
                feature_dict[f"{col}.mean"] = series.mean()
            elif stat == 'std':
                feature_dict[f"{col}.std"] = series.std()
            elif stat == 'max':
                feature_dict[f"{col}.max"] = series.max()
            elif stat == 'min':
                feature_dict[f"{col}.min"] = series.min()
            elif stat == 'mad':
                feature_dict[f"{col}.mad"] = (series - series.mean()).abs().mean()
            elif stat == 'skew':
                feature_dict[f"{col}.skew"] = series.skew()
            elif stat == 'kurt':
                feature_dict[f"{col}.kurt"] = series.kurt()
        except Exception:
            feature_dict[f"{col}.{stat}"] = np.nan
    return feature_dict

# Ana fonksiyon: Tüm araçlar için tüm özellikleri hesapla
def extract_all_features(df, vehicle_col='Vehicle_ID', sort_col='Frame_ID'):
    results = []

    if sort_col in df.columns:
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")

    for vehicle_id, group in df.groupby(vehicle_col):
        group_sorted = group.sort_values(by=sort_col).reset_index(drop=True)
        row = {vehicle_col: vehicle_id}

        for col in ['vel_diff', 'acc_diff']:
            row.update(calculate_features(group_sorted, col, stats_diff))

        for col in ['Preceding_v_Vel_Smoothed', 'Preceding_v_Acc_Smoothed']:
            row.update(calculate_features(group_sorted, col, stats_full))

        row.update(calculate_features(group_sorted, 'Preceding_Jerk', stats_full))

        results.append(row)

    return pd.DataFrame(results)

# Tüm dosyaları sırayla işle
os.makedirs(base_path, exist_ok=True)
merged_list = []

for in_path, out_path in zip(input_files, output_files):
    print(f"\İşleniyor: {in_path}")
    df = pd.read_csv(in_path)

    must_have = ['Vehicle_ID', 'Frame_ID']
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        print(f" Uyarı: Eksik temel sütun(lar): {missing} (işleme devam ediliyor)")

    segment = _segment_from_path(in_path)
    short_segment = segment.split("_")[2] + "-" + segment.split("_")[3]
    features_df = extract_all_features(df)
    features_df["Vehicle_ID"] = features_df["Vehicle_ID"].apply(lambda x: f"{short_segment}_{int(x)}")
    features_df.to_csv(out_path, index=False)
    features_df.to_excel(out_path.replace(".csv", ".xlsx"), index=False)
    print(f" Özellikler kaydedildi: {out_path}")
    merged_list.append(features_df)

#  Birleştirilmiş çıktı
merged_df = pd.concat(merged_list, ignore_index=True)
merged_df.to_csv(output_merged, index=False)
merged_df.to_excel(output_merged.replace(".csv", ".xlsx"), index=False)
print(f"\ Tüm segmentler tamamlandı. Birleşik çıktı: {output_merged}")
