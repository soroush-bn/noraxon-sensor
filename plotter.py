import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

def plot_emg_data(file_path, window_size=4000, play_all=False):
    df = pd.read_csv(file_path)
    
    time_data = df['time']
    emg_channels = [f'emg{i}' for i in range(1, 9)]
    
    num_frames = len(df) // window_size if play_all else 1
    
    for frame in range(num_frames):
        start_idx = frame * window_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df):
            end_idx = len(df)
        
        time_window = time_data[start_idx:end_idx]
        
        plt.figure(figsize=(10, 6))
        
        for channel in emg_channels:
            if channel in df.columns:
                plt.plot(time_window, df[channel][start_idx:end_idx], label=channel)
        
        plt.xlabel("Time (s)")
        plt.ylabel("EMG Signal (uV)")
        plt.title(f"EMG Data (8 Channels) - Window {frame + 1}/{num_frames}")
        plt.legend()
        
        plt.show()
        
        if play_all:
            time.sleep(0.5)
            plt.close() 

        if not play_all:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EMG data with 8 channels.")
    parser.add_argument("--file", type=str, required=True, help="Path to the merged EMG data CSV file.")
    parser.add_argument("--window_size", type=int, default=4000, help="Number of samples per window (default: 4000 for 2 seconds at 2000 Hz).")
    parser.add_argument("--play_all", action="store_true", help="Play through the entire EMG signal in windows of specified size.")
    
    args = parser.parse_args()
    
    plot_emg_data(args.file, args.window_size, args.play_all)
