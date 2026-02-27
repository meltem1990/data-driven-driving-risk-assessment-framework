
import pandas as pd
import numpy as np
import os
from scipy.stats import truncnorm

#Segment bilgileri
base_path = "D:/USveri"
file_names = [
    "trajectories-0750am-0805am_smooth_with_Gap.csv",
    "trajectories-0805am-0820am_smooth_with_Gap.csv",
    "trajectories-0820am-0835am_smooth_with_Gap.csv"
]
segment_labels = {
    "trajectories-0750am-0805am_smooth_with_Gap.csv": "0750-0805",
    "trajectories-0805am-0820am_smooth_with_Gap.csv": "0805-0820",
    "trajectories-0820am-0835am_smooth_with_Gap.csv": "0820-0835"
}

#MADR2 dağılım parametreleri
mean_dry = 8.45
std_dry = 1.40
lower_dry, upper_dry = 4.23, 12.68
a_dry = (lower_dry - mean_dry) / std_dry
b_dry = (upper_dry - mean_dry) / std_dry

# 3. Sonuçlar tutulacak liste
all_vehicle_list = []

# Her segment için işlem
for file in file_names:
    print(f" Segment işleniyor: {file}")
    segment = segment_labels[file]
    file_path = os.path.join(base_path, file)
    
    df = pd.read_csv(file_path)
    
    # Araç ID'lerini benzersiz yap
    df['Vehicle_ID'] = segment + "_" + df['Vehicle_ID'].astype(str)
    df['Preceeding'] = df['Preceeding'].astype(str)
    df.loc[df['Preceeding'] != '0', 'Preceeding'] = segment + "_" + df['Preceeding']

    # Öndeki araç bilgisi eşleştirme
    lead_info = df[['Frame_ID', 'Vehicle_ID', 'v_Vel_Smoothed']].copy()
    lead_info = lead_info.rename(columns={'Vehicle_ID': 'Preceeding', 'v_Vel_Smoothed': 'Lead_v_Vel_Smoothed'})
    df = df.merge(lead_info[['Frame_ID', 'Preceeding', 'Lead_v_Vel_Smoothed']], on=['Frame_ID', 'Preceeding'], how='left')
    
    # Göreli hız
    df['Relative_Speed'] = df['v_Vel_Smoothed'] - df['Lead_v_Vel_Smoothed']
    
    # TTC hesaplama
    df['TTC'] = np.where(
        (df['Relative_Speed'] <= 0) | (df['Gap'] <= 0) | (df['Gap'].isna()),
        100,
        df['Gap'] / df['Relative_Speed']
    )

    # TIT katkı hesaplaması
    for s in [2, 3, 4]:
        df[f'TTC_diff_{s}s'] = (s - df['TTC']).clip(lower=0)
        df[f'TTC_contrib_{s}s'] = df[f'TTC_diff_{s}s'] * 0.1

    # DRAC hesaplama
    mask = (df['Relative_Speed'] > 0) & (df['Gap'] > 0)
    df['DRAC'] = 0.0
    df.loc[mask, 'DRAC'] = 0.3048 * (df.loc[mask, 'Relative_Speed']**2) / df.loc[mask, 'Gap']

    # MADR2: araç bazlı, rastgele
    vehicle_ids = df['Vehicle_ID'].unique()
    madr2_values = truncnorm.rvs(a=a_dry, b=b_dry, loc=mean_dry, scale=std_dry, size=len(vehicle_ids), random_state=42)
    madr2_dict = dict(zip(vehicle_ids, madr2_values))
    df['MADR2'] = df['Vehicle_ID'].map(madr2_dict)

    # MADR1 sabit
    df['MADR1'] = 3.924

    # Risk etiketleri
    df['Risk_MADR1'] = (df['DRAC'] > df['MADR1']).astype(int)
    df['Risk_MADR2'] = (df['DRAC'] > df['MADR2']).astype(int)

    # TIT toplama 
    tit_df = df.groupby('Vehicle_ID').agg({
        'TTC_contrib_2s': 'sum',
        'TTC_contrib_3s': 'sum',
        'TTC_contrib_4s': 'sum'
    }).reset_index().rename(columns={
        'TTC_contrib_2s': 'TIT_t1',
        'TTC_contrib_3s': 'TIT_t2',
        'TTC_contrib_4s': 'TIT_t3'
    })

    # CPI hesaplama fonksiyonu
    def compute_cpi(group):
        sub = group[['Risk_MADR1', 'Risk_MADR2']]
        duration = len(sub)
        if duration == 0:
            return pd.Series({
                'risk_madr1_sum': 0, 'risk_madr2_sum': 0, 'duration': 0,
                'CPI_MADR1': 0, 'CPI_MADR2': 0
            })
        return pd.Series({
            'risk_madr1_sum': sub['Risk_MADR1'].sum(),
            'risk_madr2_sum': sub['Risk_MADR2'].sum(),
            'duration': duration,
            'CPI_MADR1': sub['Risk_MADR1'].sum() / duration,
            'CPI_MADR2': sub['Risk_MADR2'].sum() / duration
        })

    # CPI uygulama
    cpi_df = df.groupby('Vehicle_ID', group_keys=False)[['Risk_MADR1', 'Risk_MADR2']].apply(compute_cpi).reset_index()

    # TIT + CPI birleştir
    final_segment_df = pd.merge(
        tit_df,
        cpi_df[['Vehicle_ID', 'CPI_MADR1', 'CPI_MADR2']],
        on='Vehicle_ID'
    )

    #  TTC çıktısını kaydet
    ttc_df = df[['Vehicle_ID', 'Frame_ID', 'TTC']].copy()
    ttc_output_csv = os.path.join(base_path, f"TTC_{segment}_FrameLevel.csv")
    ttc_df.to_csv(ttc_output_csv, index=False)
    print(f" {segment} için TTC Frame-Level CSV kaydedildi: {ttc_output_csv}")

    #  Araç bazlı özet listesine eklenmesi
    all_vehicle_list.append(final_segment_df)

# Araç bazlı özetleri birleştir ve Excel olarak kaydet
all_vehicles = pd.concat(all_vehicle_list, ignore_index=True)
output_excel = os.path.join(base_path, "YENİMADR13_942_VehicleSummary.xlsx")
all_vehicles.to_excel(output_excel, sheet_name='All_Vehicle_Summary', index=False)
print(f" Vehicle-level Excel kaydedildi: {output_excel}")

