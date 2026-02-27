
import pandas as pd
import numpy as np
import os

base_path = r"D:\USveri"
input_files = [
    r"D:\USveri\jerk_output_0750_0805_with_preceding_enriched_exactonly.csv",
    r"D:\USveri\jerk_output_0805_0820_with_preceding_enriched_exactonly.csv",
    r"D:\USveri\jerk_output_0820_0835_with_preceding_enriched_exactonly.csv"
]

output_files = [
    r"D:\USveri\vehicle_features_0750_0805.xlsx",
    r"D:\USveri\vehicle_features_0805_0820.xlsx",
    r"D:\USveri\vehicle_features_0820_0835.xlsx"
]
# Kullanılacak istatistik listeleri
stats_basic = ['kurt', 'mad', 'max', 'mean', 'min',
               'p01', 'p05', 'p95', 'p99',
               'q1', 'q2', 'q3', 'skew', 'std']

stats_jerk = ['max', 'mean', 'min',
              'p01', 'p05', 'p95', 'p99',
              'q1', 'q2', 'q3', 'std']

stats_lane = ['mean', 'std', 'rng']

# Ozellik çıkarma fonksiyonu
def calculate_features(df, col, stats):
    feature_dict = {}
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
            elif stat == 'rng':
                feature_dict[f"{col}.rng"] = series.max() - series.min()
        except Exception:
            feature_dict[f"{col}.{stat}"] = np.nan
    return feature_dict

# özellik çıkarma fonksiyonu
def extract_basic_features_with_jerk_and_lane(df,
                                              vehicle_col='Vehicle_ID',
                                              sort_col='Frame_ID',
                                              basic_cols=['v_Vel_Smoothed', 'v_Acc_Smoothed', 'Gap'],
                                              jerk_col='Jerk',
                                              lateral_col='Local_X',
                                              lane_col='Lane_ID'):
    results = []
       for vehicle_id, group in df.groupby(vehicle_col):
        group_sorted = group.sort_values(by=sort_col).reset_index(drop=True)
        row = {vehicle_col: vehicle_id}
        for col in basic_cols:
            if col in group_sorted.columns:
                row.update(calculate_features(group_sorted, col, stats_basic))
        if jerk_col in group_sorted.columns:
            row.update(calculate_features(group_sorted, jerk_col, stats_jerk))
        if lateral_col in group_sorted.columns:
            try:
                row[f"{lateral_col}.std"] = group_sorted[lateral_col].std()
            except:
                row[f"{lateral_col}.std"] = np.nan
        if lane_col in group_sorted.columns:
            row.update(calculate_features(group_sorted, lane_col, stats_lane))
        results.append(row)
    return pd.DataFrame(results)

# Tüm dosyaları sırayla işle
all_features = []
for in_path, out_path in zip(input_files, output_files):
    print(f"\ İşleniyor: {in_path}")
    df = pd.read_csv(in_path)
    # Segment etiketi
    short_segment = os.path.basename(in_path).split("_")[2] + "-" + os.path.basename(in_path).split("_")[3]
    # Ozellik çıkarımı
    features_df = extract_basic_features_with_jerk_and_lane(df)
    features_df["Vehicle_ID"] = features_df["Vehicle_ID"].apply(lambda x: f"{short_segment}_{int(x)}")

    # Segment bazlı kayıt 
    features_df.to_excel(out_path, index=False)
    print(f" Özellikler kaydedildi: {out_path}")

    all_features.append(features_df)

# Tüm segmentlerin birleşimi 
merged_path = os.path.join(base_path, "vehicle_features_ALLSEGMENTS.xlsx")
merged_df = pd.concat(all_features, ignore_index=True)
merged_df.to_excel(merged_path, index=False)
print(f"\Tüm segmentler tamamlandı. Birleşik çıktı: {merged_path}")
