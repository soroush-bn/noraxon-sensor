import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import yaml
import os 
from mysignal import Signal
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")

if directory and not os.path.exists(directory):
    os.makedirs(directory)
label_maps = {"nan":"nan","Thumb Extension": "TE","index Extension":"IE","Middle Extension":"ME","Ring Extension":"RE",
             "Pinky Extension":"PE","Thumbs Up":"TU","Right Angle":"RA","Peace":"P","OK":"OK","Horn":"H","Hang Loose":"HL",
             "Power Grip":"PG","Hand Open":"HO","Wrist Extension":"WE","Wrist Flexion":"WF","Ulnar deviation":"UD","Radial Deviation":"RD","rest":"rest"}


def downsample(data, k=10):
    # Ensure the data is sorted by time
    data = data.sort_values(by='time')
    
    # Group by chunks of size k and calculate the mean
    downsampled = data.groupby(data.index // k).mean()
    
    # Keep the first time value for each group
    downsampled['time'] = data['time'].iloc[::k].reset_index(drop=True)
    downsampled['label'] = data['label'].iloc[::k].reset_index(drop=True)

    
    # Reorder the columns to place 'time' first
    columns = ['time'] + [col for col in downsampled.columns if col != 'time'] 
    downsampled = downsampled[columns]
    
    return downsampled


def show_plot(df,time_data,start_idx,end_idx,channel_pairs,indices=None):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axes = axes.flatten()  
            time_window = time_data[start_idx:end_idx]
            for ax, channels in zip(axes, channel_pairs):
                for channel in channels:
                    if channel in df.columns:
                        ax.plot(time_window, df[channel][start_idx:end_idx], label=channel)
                ax.set_ylabel("EMG Signal (uV)")
                ax.set_title(f"EMG Channels {channels[0][-1]} and {channels[1][-1]}")
                ax.legend()
                if indices is not  None:
                    for idx in indices:
                        if start_idx <= idx < end_idx: 
                            ax.axvline(x=time_data[idx], color='red', linestyle='--', linewidth=0.8)
                            ax.text(
                                time_data[idx], 
                                ax.get_ylim()[1] * 0.9, 
                                f"{label_maps[df.loc[idx, 'label']]}", 
                                rotation=0, 
                                verticalalignment='bottom', 
                                fontsize=8, 
                                color='red'
                            )
            for ax in axes[2:]:
                ax.set_xlabel("Time (s)")
            plt.tight_layout()
            plt.show()

def plot_emg_data(df, window_size=400, play_all=False,show_label= True):
    df = df.reset_index()
    df= downsample(df) #???? side effect bad
    time_data = df['time']
    emg_channels_pairs = [
            [f'emg1', f'emg2'],
            [f'emg3', f'emg4'],
            [f'emg5', f'emg6'],
            [f'emg7', f'emg8']
        ]
    if show_label:
        change_indices = df[df["label"] != df["label"].shift()].index
        start_idx = change_indices[0]
        end_idx = change_indices[-1]  # Use a wider range of data (i to i+2)
        show_plot(df,time_data,start_idx,end_idx,emg_channels_pairs,change_indices)
        plt.pause(0.5)
        plt.close()
    else :
        num_frames = len(df) // window_size if play_all else 1
        for frame in range(num_frames):
            start_idx = frame * window_size
            end_idx = start_idx + window_size
            if end_idx > len(df):
                end_idx = len(df)
            show_plot(df,time_data,start_idx,end_idx,emg_channels_pairs)
            if play_all:
                plt.pause(0.5)
                plt.close()
            if not play_all:
                break

def show_imu_plot():
    pass

def plot_imu_data(df, window_size=4000, play_all=False,show_label = True):
    df = df.reset_index()
    df= downsample(df) #???? side effect bad
    time_data = df['time']
    # print(len(df.columns))
    # Define sensor groups
    sensor_groups = {
        "accel": [f"sensor{i}_accel_{axis}" for i in range(1, 9) for axis in ["x", "y", "z"]],
        "gyro": [f"sensor{i}_gyro_{axis}" for i in range(1, 9) for axis in ["x", "y", "z"]],
        "mag": [f"sensor{i}_mag_{axis}" for i in range(1, 9) for axis in ["x", "y", "z"]]
    }
    
    if show_label:
        change_indices = df[df["label"] != df["label"].shift()].index
        start_idx = change_indices[0]
        end_idx = change_indices[-1] 
        time_window = time_data[start_idx:end_idx]

        for sensor_type, columns in sensor_groups.items():
            fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
            axes = axes.flatten()  # Flatten for easy indexing
            for i in range(8):  # Iterate over 8 sensors
                ax = axes[i]
                sensor_columns = columns[i * 3:(i + 1) * 3]  # Get x, y, z columns for the sensor
                for col in sensor_columns:
                    if col in df.columns:
                        ax.plot(time_window, df[col][start_idx:end_idx], label=col.split('_')[-1].upper())
                
                ax.set_title(f"Sensor {i + 1} - {sensor_type.capitalize()}")
                ax.set_ylabel(f"{sensor_type.capitalize()} Signal")
                ax.legend()
                if change_indices is not  None:
                    for idx in change_indices:
                        if start_idx <= idx < end_idx: 
                            ax.axvline(x=time_data[idx], color='red', linestyle='--', linewidth=0.8)
                            ax.text(
                                time_data[idx], 
                                ax.get_ylim()[1] * 0.9, 
                                f"{label_maps[df.loc[idx, 'label']]}", 
                                rotation=0, 
                                verticalalignment='bottom', 
                                fontsize=8, 
                                color='red'
                            )
        plt.tight_layout()
        plt.show()
    else:
        num_frames = len(df) // window_size if play_all else 1
        for frame in range(num_frames):
            start_idx = frame * window_size
            end_idx = start_idx + window_size

            if end_idx > len(df):
                end_idx = len(df)

            time_window = time_data[start_idx:end_idx]

            for sensor_type, columns in sensor_groups.items():
                fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
                axes = axes.flatten()  # Flatten for easy indexing

                for i in range(8):  # Iterate over 8 sensors
                    ax = axes[i]
                    sensor_columns = columns[i * 3:(i + 1) * 3]  # Get x, y, z columns for the sensor
                    
                    for col in sensor_columns:
                        if col in df.columns:
                            ax.plot(time_window, df[col][start_idx:end_idx], label=col.split('_')[-1].upper())
                    
                    ax.set_title(f"Sensor {i + 1} - {sensor_type.capitalize()}")
                    ax.set_ylabel(f"{sensor_type.capitalize()} Signal")
                    ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EMG data with 8 channels.")
    parser.add_argument("--sensor_numb", type=str, default= 1, help="to compare all data of a sensor")
    
    parser.add_argument("--window_size", type=int, default=4000, help="Number of samples per window (default: 4000 for 2 seconds at 2000 Hz).")
    parser.add_argument("--play_all", action="store_true", help="Play through the entire EMG signal in windows of specified size.")
    parser.add_argument("--plot_imu", action="store_true", help="Plot IMU data instead of EMG data.")

    args = parser.parse_args()
    df = pd.read_csv(os.path.join(directory,"final_df.csv"))
    signal_processor = Signal(config["emg_frequency"])
    
    if args.plot_imu:
        print("plotting imu")
        dfs_gestures = signal_processor.get_gestures_dataframes(df)
        for i in range(len(dfs_gestures)):
            plot_imu_data(dfs_gestures[i], args.window_size, args.play_all,True)
    else :
        print("ploting emg")
        dfs_gestures = signal_processor.get_gestures_dataframes(df)
        for i in range(len(dfs_gestures)):
            plot_emg_data(dfs_gestures[i], args.window_size, args.play_all,show_label= True)