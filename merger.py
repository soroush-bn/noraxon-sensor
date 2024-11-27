
import os 
import pandas as pd 
import re
import argparse
import numpy as np 
class Merger():
    def __init__(self,folder_path , emg_path,imu_path) :
        self.folder_path= folder_path
        self.emg_path = emg_path
        self.imu_path = imu_path



    def merge_emg_data(self):
        merged_data = {}
        
        for i in range(1, 9):
            filename = f"Ultium_EMG-Ultium_EMG.EMG_{i}.csv"
            file_path = os.path.join(self.folder_path, filename)
            
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path, skiprows=2)
                
                time_column = df['time']
                value_column = df['value']
                
                if 'time' not in merged_data:
                    merged_data['time'] = time_column
                
                merged_data[f'emg{i}'] = value_column

        merged_df = pd.DataFrame(merged_data)
        merged_df.to_csv(self.emg_path, index=False)
        print(f"Merged CSV saved to {self.emg_path}")
        return merged_df

        
    def merge_imu_data(self):
        merged_data = {}
        
        pattern = r"Ultium_EMG-Ultium_EMG\.Internal_(Accel|Gyro|Mag)_(\d)_([AGM])([xyz])\.csv"
        
        for filename in os.listdir(self.folder_path):
            match = re.match(pattern, filename)
            if match:
                data_type = match.group(1).lower()  
                sensor_number = match.group(2)       
                axis = match.group(4).lower()        
                
                column_name = f"sensor{sensor_number}_{data_type}_{axis}"
                
                file_path = os.path.join(self.folder_path, filename)
                
                df = pd.read_csv(file_path, skiprows=2)
                
                time_column = df['time']
                value_column = df['value']
                
                if 'time' not in merged_data:
                    merged_data['time'] = time_column
                
                merged_data[column_name] = value_column

        merged_df = pd.DataFrame(merged_data)
        
        merged_df.to_csv(self.imu_path, index=False)
        print(f"Merged IMU CSV saved to {self.imu_path}")
        return merged_df

        
    def get_same_freq(self,imu_df, emg_df, imu_freq, emg_freq, method="repeat"):
        ratio = emg_freq // imu_freq  # The ratio of frequencies (2000/200 = 10)
        
        if method == "repeat":
            # Repeat IMU rows to match EMG frequency
            imu_expanded = imu_df.reindex(imu_df.index.repeat(ratio)).reset_index(drop=True)
            imu_expanded["time"] = np.linspace(imu_df["time"].iloc[0], imu_df["time"].iloc[-1], len(imu_expanded))
            merged_df = pd.concat([imu_expanded, emg_df.iloc[:, 1:].reset_index(drop=True)], axis=1)
        
        elif method == "average":
            # Average EMG rows to match IMU frequency
            emg_downsampled = emg_df.groupby(emg_df.index // ratio).mean().reset_index(drop=True)
            emg_downsampled["time"] = imu_df["time"]  # Use IMU time for consistency
            merged_df = pd.concat([imu_df.reset_index(drop=True), emg_downsampled.iloc[:, 1:]], axis=1)
        
        else:
            raise ValueError("Invalid method. Choose 'repeat' or 'average'.")
        
        return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge IMU&EMG data files.")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing IMU data CSV files.")
    parser.add_argument("--imu_path", type=str, default="merged_imu_data.csv", help="Output file name for the merged imu CSV (default: merged_imu_data.csv)")
    parser.add_argument("--emg_path", type=str, default="merged_emg_data.csv", help="Output file name for merged emg csv")
    parser.add_argument("--final_path", type=str, default="merged.csv", help="Output file name for merged emg csv")
    
    args = parser.parse_args()
    
    merger = Merger(folder_path=args.path ,imu_path=args.imu_path , emg_path = args.emg_path )
    final_df = merger.get_same_freq(merger.merge_imu_data(),merger.merge_emg_data(),200,2000,"average")
    final_df.to_csv(args.final_path, index=False)

    print(final_df.head())
    print(len(final_df))

    
