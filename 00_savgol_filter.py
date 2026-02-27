
import pandas as pd
from scipy.signal import savgol_filter

def apply_sg_filter(input_path, output_base_path, window_length=11):
    df = pd.read_csv(input_path)
    df = df.sort_values(by=['Vehicle_ID', 'Frame_ID'])

    filtered_list = []

    for vehicle_id, group in df.groupby('Vehicle_ID'):
        group = group.copy()
        group = group.dropna(subset=['v_Vel', 'v_Acc'])

        speed_series = group['v_Vel'].values
        accel_series = group['v_Acc'].values
        is_zero = (speed_series == 0) & (accel_series == 0)

        if len(group) >= window_length:
            speed_smooth = savgol_filter(speed_series, window_length=window_length, polyorder=1)
            accel_smooth = savgol_filter(accel_series, window_length=window_length, polyorder=2)
            speed_smooth[is_zero] = speed_series[is_zero]
            accel_smooth[is_zero] = accel_series[is_zero]
        else:
            speed_smooth = speed_series
            accel_smooth = accel_series

        group['v_Vel_Smoothed'] = speed_smooth
        group['v_Acc_Smoothed'] = accel_smooth
        filtered_list.append(group)

    df_filtered = pd.concat(filtered_list, ignore_index=True)

    excel_path = output_base_path + ".xlsx"
    csv_path = output_base_path + ".csv"
    
    df_filtered.to_excel(excel_path, index=False)
    df_filtered.to_csv(csv_path, index=False)

    print(f" SG filtresi uygulandı ve kayıt edildi:\n - {excel_path}\n - {csv_path}")
    
apply_sg_filter(
    input_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0750am-0805am.csv",
    output_base_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0750am-0805am_smooth"
)

apply_sg_filter(
    input_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0805am-0820am.csv",
    output_base_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0805am-0820am_smooth"
)

apply_sg_filter(
    input_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0820am-0835am.csv",
    output_base_path="C:/Users/aslantas/Desktop/guncelveri/trajectories-0820am-0835am_smooth"
)
