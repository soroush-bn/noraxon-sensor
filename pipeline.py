import pandas as pd
import numpy as np
from mysignal import Signal
import yaml 
from segmentation import Segmentation
from feature_extraction import Feature
from classification import train_LDA,train_svm
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if __name__=="__main__":
    df = pd.read_csv(r"E:\projects\noraxon\noraxon-sensor\data\so_Baghernezhad_second\final_df.csv")
    sampling_rate = config["emg_frequency"]
    print(df)
    signal_processor = Signal(sampling_rate)
    signal_processor.calculate_per_gesture(df)
    dfs_gestures = signal_processor.get_gestures_dataframes(df)

    feature = Feature()
    gesture_dfs = []

    # Step 2: Iterate over each gesture
    names = df.columns[2:-2]  # Get channel names (e.g., Channel1, Channel2, etc.)
    for gesture_df in dfs_gestures:
        
        segmentation = Segmentation(gesture_df, config["emg_frequency"], config["window_size"], config["overlapping"])
        segmented_dfs = segmentation.segment()  # Get segmented DataFrames

        all_channels_features = []  # Store the features for all segments of this gesture
        for df_segment in segmented_dfs:  # Iterate through each segmented window of the gesture
            channel_features = []  # Store features for this window
            for channel in names:  # Loop through each channel
                # Extract features for the current channel in the current segment
                time_features = feature.get_time_features(df_segment[channel].values)  # Extract time features
                freq_features = feature.get_freq_featrures(df_segment[channel].values)  # Extract frequency features
                
                # Concatenate time-domain and frequency-domain features
                channel_features.extend(time_features)  # Add time features
                channel_features.extend(freq_features)  # Add frequency features

            all_channels_features.append(channel_features)  # Store the features for this window

        # Step 3: Create feature column names for each channel
        feature_columns = []
        for channel in names:
            for feature_name in feature.time_domain_features:
                feature_columns.append(f"{channel}_{feature_name}")  # E.g., "Channel1_MAV", "Channel2_VAR"
            for feature_name in feature.frequency_domain_features:
                feature_columns.append(f"{channel}_{feature_name}")  # E.g., "Channel1_MNF", "Channel2_MDF"

        # Step 4: Create the DataFrame for this gesture
        gesture_features_df = pd.DataFrame(all_channels_features, columns=feature_columns)
        gesture_dfs.append(gesture_features_df)  # Store the DataFrame for this gesture
    gesture_dfs
    print(gesture_dfs[1].head())

    train_LDA(gesture_dfs)
    train_svm(gesture_dfs)
    print("done")