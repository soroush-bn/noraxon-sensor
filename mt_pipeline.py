import pandas as pd
import numpy as np
from mysignal import Signal
import yaml
from segmentation import Segmentation
from feature_extraction import Feature
from classification import train_LDA, train_svm
import os
from tqdm import tqdm
from time import time
from multiprocessing import Pool, cpu_count

def process_channel(channel, df_segment, feature):
    """Extract features for a single channel."""
    time_features = feature.get_time_features(df_segment[channel].values)
    freq_features = feature.get_freq_featrures(df_segment[channel].values)
    return time_features + freq_features

def process_gesture(gesture_df, config, feature, names):
    """Process a single gesture dataframe."""
    segmentation = Segmentation(gesture_df, config["emg_frequency"], config["window_size"], config["overlapping"])
    segmented_dfs = segmentation.segment()  # Get segmented DataFrames

    all_channels_features = []

    # Use multiprocessing to process each segment
    for df_segment in tqdm(segmented_dfs, desc="Segmenting Windows", unit="window", leave=False):
        with Pool(cpu_count()-2) as pool:
            channel_features = pool.starmap(
                process_channel,
                [(channel, df_segment, feature) for channel in names]
            )
        # Flatten the list of features and append
        all_channels_features.append([item for sublist in channel_features for item in sublist])

    return all_channels_features

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, "final_df.csv")
    df = pd.read_csv(path)
    sampling_rate = config["emg_frequency"]
    print(df)

    signal_processor = Signal(sampling_rate)
    dfs_gestures = signal_processor.get_gestures_dataframes(df)

    signal_processor.calculate_per_gesture(dfs_gestures)

    feature = Feature()
    gesture_dfs = []

    names = df.columns[2:-3]  # Get channel names (e.g., Channel1, Channel2, etc.)
    if config["load_features"]==True:
        for n in range(config["number_of_gestures"]):
            try:
                df = pd.read_csv(os.path.join(directory,f'gesture_features{n}.csv'))
                gesture_dfs.append(df)
            except: 
                break
    else:
    # Parallelize gesture processing
        for gesture_df in tqdm(dfs_gestures, desc="Processing Gestures", unit="gesture"):
            gesture_features = process_gesture(gesture_df, config, feature, names)

            # Create DataFrame for gesture features
            feature_columns = []
            for channel in names:
                for feature_name in feature.time_domain_features:
                    feature_columns.append(f"{channel}_{feature_name}")
                for feature_name in feature.frequency_domain_features:
                    feature_columns.append(f"{channel}_{feature_name}")

            gesture_features_df = pd.DataFrame(gesture_features, columns=feature_columns)
            gesture_dfs.append(gesture_features_df)  # Store the DataFrame for this gesture

    train_LDA(gesture_dfs)
    train_svm(gesture_dfs)

    for i in range(len(gesture_dfs)):
        gesture_dfs[i].to_csv(os.path.join(directory, f"gesture_features{i}.csv"))

    print("done")
