import pandas as pd
import os
import argparse
import re

def merge_imu_data(folder_path, output_path="merged_imu_data.csv"):
    merged_data = {}
    
    pattern = r"Ultium_EMG-Ultium_EMG\.Internal_(Accel|Gyro|Mag)_(\d)_([AGM])([xyz])\.csv"
    
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            data_type = match.group(1).lower()  
            sensor_number = match.group(2)       
            axis = match.group(4).lower()        
            
            column_name = f"sensor{sensor_number}_{data_type}_{axis}"
            
            file_path = os.path.join(folder_path, filename)
            
            df = pd.read_csv(file_path, skiprows=2)
            
            time_column = df['time']
            value_column = df['value']
            
            if 'time' not in merged_data:
                merged_data['time'] = time_column
            
            merged_data[column_name] = value_column

    merged_df = pd.DataFrame(merged_data)
    
    merged_df.to_csv(output_path, index=False)
    print(f"Merged IMU CSV saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge IMU data files.")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing IMU data CSV files.")
    parser.add_argument("--output", type=str, default="merged_imu_data.csv", help="Output file name for the merged CSV (default: merged_imu_data.csv)")
    
    args = parser.parse_args()
    
    merge_imu_data(args.path, args.output)
