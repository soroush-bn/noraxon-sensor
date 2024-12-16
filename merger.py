import yaml
import os 
import pandas as pd 
import re
import argparse
import numpy as np 

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")
 
if directory and not os.path.exists(directory):
    os.makedirs(directory)

class Merger():
    def __init__(self,folder_path ) :
        self.folder_path= folder_path
        self.emg_path = "merged_emg_data.csv"
        self.imu_path = "merged_imu_data.csv"



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
    
    def add_time_stamps(self, final_df):
        text_dir = directory + "/" + config["first_name"] + config["last_name"] + "_timestamps.txt"
        with open(text_dir, "r") as f:
            first_timestamp, last_timestamp = map(float, f.read().strip().split(","))
        total_rows = len(final_df)
        interval = (last_timestamp - first_timestamp) / (total_rows - 1)
        final_df['timestamp'] = [first_timestamp + i * interval for i in range(total_rows)]
        merged_df=final_df
        print("time stamps added")
        return merged_df

        

    def merge_with_labels(self,merged_df,label_df):
        first_merged_df_ts = merged_df.loc[0,"timestamp"]
        # first_label_df_ts = label_df.loc[0,"time_stamp"]
        names = ["Thumb Extension","index Extension","Middle Extension","Ring Extension",
             "Pinky Extension","Thumbs Up","Right Angle","Peace","OK","Horn","Hang Loose",
             "Power Grip","Hand Open","Wrist Extension","Wrist Flexion","Ulnar deviation","Radial Deviation"]
    
        merged_df['label'] = ["None" for i in range(len(merged_df))]
        sw= False
        # for i in range(0,len(merged_df),config["IMU_frequency"]):
        i=0 
        labels = []
        for g in range(len(names)):
            for r in range(config["number_of_gesture_repetition"]):
                labels.append(names[g])
                labels.append("rest")

        c=0 
        for i in range(len(merged_df)):
            merged_df.loc[i,"label"]  =labels[c]
            if i% config["IMU_frequency"]==0 and i!=0:
                c+=1 
             
        print("merged with labels")
        return merged_df 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge IMU&EMG data files.")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing IMU data CSV files.")
    # parser.add_argument("--imu_path", type=str, default="merged_imu_data.csv", help="Output file name for the merged imu CSV (default: merged_imu_data.csv)")
    # parser.add_argument("--emg_path", type=str, default="label_data.csv", help="Output file name for merged emg csv")
    parser.add_argument("--label_path", type=str, default="merged_emg_data.csv", help="Output file name for merged emg csv")
    
    # parser.add_argument("--final_path", type=str, default="merged.csv", help="Output file name for merged emg csv")
    
    args = parser.parse_args()
    
    merger = Merger(folder_path=args.path )
    print("0")
    
    final_df = merger.get_same_freq(merger.merge_imu_data(),merger.merge_emg_data(),200,2000,"average")
    print("1")
    final_df = merger.add_time_stamps(final_df)
    # label_df = pd.read_csv(args.label_path)
    print("2")
    
    final_df = merger.merge_with_labels(final_df,final_df)
    final_df_dir = os.path.join(directory,"final_df.csv")
    final_df.to_csv(final_df_dir)

    print(final_df.head())
    print(len(final_df))

    
#todo having 