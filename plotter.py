import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

def plot_emg_data(file_path, window_size=4000, play_all=False):
    df = pd.read_csv(file_path)
    
    time_data = df['time']
    emg_channels_pairs = [
        [f'emg1', f'emg2'],
        [f'emg3', f'emg4'],
        [f'emg5', f'emg6'],
        [f'emg7', f'emg8']
    ]
    
    num_frames = len(df) // window_size if play_all else 1
    
    for frame in range(num_frames):
        start_idx = frame * window_size
        end_idx = start_idx + window_size
        
        if end_idx > len(df):
            end_idx = len(df)
        
        time_window = time_data[start_idx:end_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        
        # Plot each pair of EMG channels in separate subplots
        for i, (ax, channels) in enumerate(zip(axes, emg_channels_pairs)):
            for channel in channels:
                if channel in df.columns:
                    ax.plot(time_window, df[channel][start_idx:end_idx], label=channel)
            ax.set_ylabel("EMG Signal (uV)")
            ax.set_title(f"EMG Channels {channels[0][-1]} and {channels[1][-1]}")
            ax.legend()
        
        # Add the x-axis label to the bottom subplots only
        for ax in axes[2:]:
            ax.set_xlabel("Time (s)")
        
        plt.tight_layout()
        plt.show()
        
        if play_all:
            time.sleep(0.5)  # Adjust this to control the speed between windows
            plt.close()  # Close the plot to avoid overlapping if in play mode

        # Break the loop if only one window is to be shown
        if not play_all:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EMG data with 8 channels.")
    parser.add_argument("--file", type=str, required=True, help="Path to the merged EMG data CSV file.")
    parser.add_argument("--window_size", type=int, default=4000, help="Number of samples per window (default: 4000 for 2 seconds at 2000 Hz).")
    parser.add_argument("--play_all", action="store_true", help="Play through the entire EMG signal in windows of specified size.")
    
    args = parser.parse_args()
    
    plot_emg_data(args.file, args.window_size, args.play_all)
