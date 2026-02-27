

import pandas as pd
import os
import numpy as np

base_path = "C:/Users/aslantas/Desktop/guncelveri"
file_names = [
    "trajectories-0750am-0805am_smooth.csv",
    "trajectories-0805am-0820am_smooth.csv",
    "trajectories-0820am-0835am_smooth.csv"
]

for idx, file_name in enumerate(file_names):
    print(f" İşleniyor: {file_name}")
    file_path = os.path.join(base_path, file_name)
    df = pd.read_csv(file_path)

    #  Lead araç bilgileri (aynı frame'deki öndeki araç bilgisi)
    lead_info = df[['Frame_ID', 'Vehicle_ID', 'Local_Y', 'v_Length']].copy()
    lead_info = lead_info.rename(columns={
        'Vehicle_ID': 'Preceeding',
        'Local_Y': 'Lead_Y',
        'v_Length': 'Lead_Length'
    })
    lead_info = lead_info.drop_duplicates(subset=['Frame_ID', 'Preceeding'])

    #  Birleşriem işlemi
    df = df.merge(
        lead_info[['Frame_ID', 'Preceeding', 'Lead_Y', 'Lead_Length']],
        on=['Frame_ID', 'Preceeding'],
        how='left'
    )

    # 4. GAP hesaplama
    def compute_gap(row):
        if row['Preceeding'] == 0 or pd.isna(row['Lead_Y']) or pd.isna(row['Lead_Length']):
            return 0
        return (row['Lead_Y'] - row['Lead_Length']) - row['Local_Y']

    df['Gap'] = df.apply(compute_gap, axis=1)

    # GapRevize = Space_Headway - Lead_Length
    if 'Space_Headway' in df.columns and 'Lead_Length' in df.columns:
        df['GapRevize'] = df['Space_Headway'] - df['Lead_Length']
        df.loc[df['GapRevize'] < 0, 'GapRevize'] = np.nan

    #  CSV olarak kaydet (her segment için)
    output_csv = os.path.join(base_path, file_name.replace(".csv", "_with_Gap.csv"))
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV kaydedildi: {output_csv}")

    #  İlk segment için ayrıca Excel çıktısı oluştur
    if idx == 0:
        output_excel = os.path.join(base_path, file_name.replace(".csv", "_with_Gap.xlsx"))
        df.to_excel(output_excel, index=False)
        print(f"📁 Excel de kaydedildi (ilk segment): {output_excel}")

print(" Tüm segmentlerde Gap ve GapRevize hesaplandı. İlk segment Excel olarak kaydedildi.")
