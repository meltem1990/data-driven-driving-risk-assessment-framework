
import pandas as pd

input_files = [
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\trajectories-0750am-0805am_smooth_with_Gap.csv",
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\trajectories-0805am-0820am_smooth_with_Gap.csv",
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\trajectories-0820am-0835am_smooth_with_Gap.csv"
]

output_files = [
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\jerk_output_0750_0805.csv",
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\jerk_output_0805_0820.csv",
    "C:\\Users\\aslantas\\Desktop\\guncelveri\\jerk_output_0820_0835.csv"
]

# Zaman aralığı
time_interval = 0.1

# Jerk hesaplama fonksiyonu 
def calculate_jerk(group, vehicle_id):
    group = group.sort_values(by='Frame_ID').reset_index(drop=True)
    group['Jerk'] = group['v_Acc_Smoothed'].diff() / (group['Frame_ID'].diff() * time_interval)
    group['Vehicle_ID'] = vehicle_id  
    return group

for idx, (input_path, output_path) in enumerate(zip(input_files, output_files)):
    print(f"İşleniyor: {input_path}")
    df = pd.read_csv(input_path)
    result_list = []

    for vehicle_id, group in df.groupby('Vehicle_ID'):
        jerked_group = calculate_jerk(group, vehicle_id)
        result_list.append(jerked_group)

    df_with_jerk = pd.concat(result_list, ignore_index=True)

    df_with_jerk.to_csv(output_path, index=False)
    print(f" Kaydedildi (CSV): {output_path}")
print("\Tüm segmentler başarıyla işlendi.")

