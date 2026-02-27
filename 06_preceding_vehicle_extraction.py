
import pandas as pd
import numpy as np

input_files = [
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0750_0805.csv",
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0805_0820.csv",
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0820_0835.csv"
]

output_files = [
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0750_0805_with_preceding_enriched_exactonly.csv",
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0805_0820_with_preceding_enriched_exactonly.csv",
    r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0820_0835_with_preceding_enriched_exactonly.csv"
]

for in_path, out_path in zip(input_files, output_files):
    print(f"\İşleniyor: {in_path}")
    df = pd.read_csv(in_path)

# Dosyada "Preceeding" varsa -> "Preceding" yap
if "Preceeding" in df.columns and "Preceding" not in df.columns:
    df.rename(columns={"Preceeding": "Preceding"}, inplace=True)

# Tip dönüşümleri ve temizlik
df["Frame_ID"] = pd.to_numeric(df["Frame_ID"], errors="coerce").astype("Int64")
df["Preceding"] = pd.to_numeric(df["Preceding"], errors="coerce")  

# Lookup indexi
required_cols = {"Vehicle_ID", "Frame_ID", "Local_Y", "v_Vel_Smoothed", "v_Acc_Smoothed", "Jerk", "v_Class"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Beklenen sütunlar eksik: {missing}")

df_lookup = df.set_index(["Vehicle_ID", "Frame_ID"])

#  Eşleşme fonksiyonu 
def find_preceding_values(row):
    pid = row["Preceding"]     
    fid = row["Frame_ID"]      

       if pd.isna(pid):
        return pd.Series([np.nan, np.nan, np.nan, None, np.nan, np.nan, np.nan, np.nan])

    pid = int(pid)
    key = (pid, int(fid))

    if key not in df_lookup.index:
        return pd.Series([np.nan, np.nan, np.nan, "Information not available", np.nan, np.nan, np.nan, np.nan])

    rec = df_lookup.loc[key]
    # Çoklu kayıt olursa ilkini seç
    if isinstance(rec, pd.DataFrame):
        rec = rec.iloc[0]
    y_diff = float(rec["Local_Y"]) - float(my_y) if pd.notna(my_y) and pd.notna(rec["Local_Y"]) else np.nan
    return pd.Series([
        float(rec["v_Vel_Smoothed"]),
        float(rec["v_Acc_Smoothed"]),
        int(fid),
        "Available",
        float(rec["Jerk"]),
        int(rec["v_Class"]),
        float(rec["Local_Y"]),
        y_diff
    ])

#  Fonksiyonu uygula 
df[[
    "Preceding_v_Vel_Smoothed", "Preceding_v_Acc_Smoothed",
    "Preceding_Matched_Frame_ID", "Preceding_Status",
    "Preceding_Jerk", "Preceding_Class",
    "Preceding_y", "Preceding_y_Diff"
]] = df.apply(find_preceding_values, axis=1)

# Türevlerİ
df["Frame_Diff"] = (df["Frame_ID"].astype("float") - df["Preceding_Matched_Frame_ID"].astype("float")).abs()
df["vel_diff"] = df["v_Vel_Smoothed"] - df["Preceding_v_Vel_Smoothed"]
df["acc_diff"] = df["v_Acc_Smoothed"] - df["Preceding_v_Acc_Smoothed"]

# İstatistikler
valid_preceding_count = (df["Preceding_Status"] == "Available").sum()
print(f"\Preceding verisi bulunan kayıt sayısı: {valid_preceding_count}")

missing_due_to_frame_id = df[
    (df["Preceding"].notna()) & (df["Preceding_Status"] == "Information not available")
].shape[0]
print(f" Preceding aracı olduğu halde aynı Frame_ID’de veri bulunamayan satır sayısı: {missing_due_to_frame_id}")

# kaç satırda Preceding tamamen yok 
no_preceding = df["Preceding"].isna().sum()
print(f" Önünde araç olmayan (Preceding NaN) satır: {no_preceding}")

#Kaydet
output_path = r"C:\Users\aslantas\Desktop\guncelveri\jerk_output_0750_0805_with_preceding_enriched_exactonly.csv"
df.to_csv(output_path, index=False, float_format="%.6f")
print(f"\n CSV dosyası başarıyla kaydedildi: {output_path}")
