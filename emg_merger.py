import pandas as pd
import os
import argparse

def merge_emg_data(folder_path, output_path="merged_emg_data.csv"):
    merged_data = {}
    
    for i in range(1, 9):
        filename = f"Ultium_EMG-Ultium_EMG.EMG_{i}.csv"
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, skiprows=2)
            
            time_column = df['time']
            value_column = df['value']
            
            if 'time' not in merged_data:
                merged_data['time'] = time_column
            
            merged_data[f'emg{i}'] = value_column

    merged_df = pd.DataFrame(merged_data)
    
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge EMG data files.")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing EMG data CSV files.")
    parser.add_argument("--output", type=str, default="merged_emg_data.csv", help="Output file name for the merged CSV (default: merged_emg_data.csv)")
    
    args = parser.parse_args()
    
    merge_emg_data(args.path, args.output)
