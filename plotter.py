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
        
        for i, (ax, channels) in enumerate(zip(axes, emg_channels_pairs)):
            for channel in channels:
                if channel in df.columns:
                    ax.plot(time_window, df[channel][start_idx:end_idx], label=channel)
            ax.set_ylabel("EMG Signal (uV)")
            ax.set_title(f"EMG Channels {channels[0][-1]} and {channels[1][-1]}")
            ax.legend()
        
        for ax in axes[2:]:
            ax.set_xlabel("Time (s)")
        
        plt.tight_layout()
        plt.show()
        
        if play_all:
            time.sleep(0.5)  
            plt.close()  

        if not play_all:
            break

def plot_imu_data(file_path, window_size=4000, play_all=False):
    df = pd.read_csv(file_path)
    time_data = df['time']

    num_frames = len(df) // window_size if play_all else 1

    for frame in range(num_frames):
        start_idx = frame * window_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            end_idx = len(df)

        time_window = time_data[start_idx:end_idx]

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Accelerometer plot
        axes[0].plot(time_window, df['sensor8_accel_x'][start_idx:end_idx], label='Accel X')
        axes[0].plot(time_window, df['sensor8_accel_y'][start_idx:end_idx], label='Accel Y')
        axes[0].plot(time_window, df['sensor8_accel_z'][start_idx:end_idx], label='Accel Z')
        axes[0].set_title("Accelerometer")
        axes[0].set_ylabel("Acceleration")
        axes[0].legend()

        # Gyroscope plot
        axes[1].plot(time_window, df['sensor8_gyro_x'][start_idx:end_idx], label='Gyro X')
        axes[1].plot(time_window, df['sensor8_gyro_y'][start_idx:end_idx], label='Gyro Y')
        axes[1].plot(time_window, df['sensor8_gyro_z'][start_idx:end_idx], label='Gyro Z')
        axes[1].set_title("Gyroscope")
        axes[1].set_ylabel("Angular Velocity")
        axes[1].legend()

        # Magnetometer plot
        axes[2].plot(time_window, df['sensor8_mag_x'][start_idx:end_idx], label='Mag X')
        axes[2].plot(time_window, df['sensor8_mag_y'][start_idx:end_idx], label='Mag Y')
        axes[2].plot(time_window, df['sensor8_mag_z'][start_idx:end_idx], label='Mag Z')
        axes[2].set_title("Magnetometer")
        axes[2].set_ylabel("Magnetic Field")
        axes[2].legend()

        for ax in axes:
            ax.set_xlabel("Time (s)")

        plt.tight_layout()
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
    parser.add_argument("--plot_imu", action="store_true", help="Plot IMU data instead of EMG data.")

    args = parser.parse_args()
    
    if args.plot_imu:
        plot_imu_data(args.file, args.window_size, args.play_all)
    else:
        plot_emg_data(args.file, args.window_size, args.play_all)