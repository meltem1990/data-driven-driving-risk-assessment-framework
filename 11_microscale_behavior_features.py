
import pandas as pd
import numpy as np
import os

#Parametreler
variables = ['v_Vel_Smoothed', 'Preceding_v_Vel_Smoothed',
             'v_Acc_Smoothed', 'Preceding_v_Acc_Smoothed',
             'Gap', 'vel_diff', 'acc_diff']
window_lengths = {'w1': 10, 'w2': 50, 'w3': 100}
min_vals_required = 3

base_path = r"D:\USveri"
input_files = [
    os.path.join(base_path, "jerk_output_0750_0805_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0805_0820_with_preceding_enriched_exactonly.csv"),
    os.path.join(base_path, "jerk_output_0820_0835_with_preceding_enriched_exactonly.csv")
]

def segment_from_path(path):
    match = os.path.basename(path).split("_")[2:4]
    return "-".join(match)

# Hareketli pencere değerlerini getirilmesi
def get_window_vals(frames, values, base_frame, offset):
    target = base_frame + offset
    if target not in frames:
        return np.array([])
    lower, upper = sorted([base_frame, target])
    return values[(frames >= lower) & (frames <= upper)]

#  Özellik hesaplama fonksiyonu
def extract_moving_features(df, variable, offset, label):
    results = []
    for vid, group in df.groupby('Vehicle_ID'):
        group = group.sort_values('Frame_ID')
        #frames → o aracın tüm frame numaraları (örnek [100, 101, 102, ...])
        #values → seçilen değişkenin değerleri (örnek hız [15.2, 15.5, 16.0, ...]).
        frames = group['Frame_ID'].values
        values = group[variable].values
        rng_list, crng_list, sma_list, msd_list, rsd_list, emar_list = [], [], [], [], [], []
        for i in range(len(group)):
            t = frames[i]
            window_vals = get_window_vals(frames, values, t, offset)
            window_vals = window_vals[~np.isnan(window_vals)]
            if len(window_vals) >= min_vals_required:
                rng = np.max(window_vals) - np.min(window_vals)
                denom = np.max(window_vals) + np.min(window_vals)
                crng = rng / denom if denom != 0 else np.nan
                sma = np.mean(window_vals)
                msd = np.std(window_vals)
                rsd = msd / sma if sma != 0 else np.nan
                ema = pd.Series(window_vals).ewm(span=len(window_vals), adjust=False).mean().iloc[-1]
                emar = window_vals[-1] / (window_vals[-1] - ema) if (window_vals[-1] - ema) != 0 else np.nan
            else:
                rng = crng = sma = msd = rsd = emar = np.nan

            rng_list.append(rng)
            crng_list.append(crng)
            sma_list.append(sma)
            msd_list.append(msd)
            rsd_list.append(rsd)
            emar_list.append(emar)

        feature_dict = {'Vehicle_ID': vid}
        metrics = {'rng': rng_list, 'crng': crng_list, 'sma': sma_list,
                   'msd': msd_list, 'rsd': rsd_list, 'emar': emar_list}

        for metric, values_list in metrics.items():
            series = pd.Series(values_list).dropna()
            prefix = f"{variable}.{metric}.{label}"
            feature_dict[f"{prefix}.mean"] = series.mean()
            feature_dict[f"{prefix}.std"] = series.std()
            feature_dict[f"{prefix}.max"] = series.max()
            feature_dict[f"{prefix}.min"] = series.min()

        results.append(feature_dict)

    return pd.DataFrame(results)

#Ana döngü: dosyaları oku, özellikleri hesapla, birleştir
all_features = []

for path in input_files:
    segment = segment_from_path(path)
    df = pd.read_csv(path)
    df['Vehicle_ID'] = df['Vehicle_ID'].apply(lambda x: f"{segment}_{int(x)}")

    feature_dfs = []

    for var in variables:
        for label, offset in window_lengths.items():
            df_feat = extract_moving_features(df, var, offset, label)
            feature_dfs.append(df_feat)

    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Vehicle_ID', how='outer'), feature_dfs)
    all_features.append(df_merged)

# Tüm segmentler birleştir
df_final = pd.concat(all_features, ignore_index=True)

output_path = os.path.join(base_path, "microscale_features_w1_w2_w3_ALLSEGMENTS.xlsx")
df_final.to_excel(output_path, index=False)
print("Özellikler hesaplandı ve kaydedildi:", output_path)
