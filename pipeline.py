import pandas as pd
import numpy as np
from mysignal import Signal
import yaml 
from segmentation import Segmentation
from feature_extraction import Feature
from classification import train_LDA,train_svm
import os 
from tqdm import tqdm
from time import time
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")
 
if directory and not os.path.exists(directory):
    os.makedirs(directory)
if __name__=="__main__":
    path = os.path.join(directory,"final_df.csv")
    df = pd.read_csv(path)
    sampling_rate = config["emg_frequency"]
    print(df)
    signal_processor = Signal(sampling_rate)
    dfs_gestures = signal_processor.get_gestures_dataframes(df)

    signal_processor.calculate_per_gesture(dfs_gestures)

    feature = Feature()
    gesture_dfs = []
    if config["load_features"]==True:
        for n in range(config["number_of_gestures"]-1):
                df = pd.read_csv(os.path.join(directory,f'gesture_features{n}.csv'))
                gesture_dfs.append(df)
    else:      
        columns = df.columns[2:-3]  # Get channel names (e.g., Channel1, Channel2, etc.)
        for gesture_df in tqdm(dfs_gestures, desc="Processing Gestures", unit="gesture"):   
            segmentation = Segmentation(gesture_df, config["emg_frequency"], config["window_size"], config["overlapping"])
            segmented_dfs = segmentation.segment()  # Get segmented DataFrames
            time_time_domain = []
            time_frequency_domain = [] 
            all_channels_features = []  # Store the features for all segments of this gesture
            for df_segment in tqdm(segmented_dfs, desc="Segmenting Windows", unit="window", leave=False):  

                channel_features = []  # Store features for this window
                for channel in tqdm(columns, desc="Extracting Features", unit="channel", leave=False):  
                    t_Start = time()
                    time_features = feature.get_time_features(df_segment[channel].values)  # Extract time features
                    t_end = time()
                    time_time_domain.append(t_end-t_Start)
                    freq_features = feature.get_freq_featrures(df_segment[channel].values)  # Extract frequency features
                    time_frequency_domain.append(t_end-t_Start)
                    
                    channel_features.extend(time_features)  # Add time features
                    channel_features.extend(freq_features)  # Add frequency features


                all_channels_features.append(channel_features)  # Store the features for this window
            print(f'time_domain features took: {np.sum(time_time_domain)} \n freq_domain features took: {np.sum(time_frequency_domain)}')
            feature_columns = []
            for channel in columns:
                for feature_name in feature.time_domain_features:
                    feature_columns.append(f"{channel}_{feature_name}")  # E.g., "Channel1_MAV", "Channel2_VAR"
                for feature_name in feature.frequency_domain_features:
                    feature_columns.append(f"{channel}_{feature_name}")  # E.g., "Channel1_MNF", "Channel2_MDF"

            gesture_features_df = pd.DataFrame(all_channels_features, columns=feature_columns)
            gesture_dfs.append(gesture_features_df)  # Store the DataFrame for this gesture
    #table6
    table6 = []
    table6_df= pd.DataFrame()  
       
    for f in feature.time_domain_features + feature.frequency_domain_features: #this shoud be among all participant cause
        # only 4 features is not enough for good accuracy
        accuracy_lda_wrist,_ = train_LDA(gesture_dfs,f"wrist_{f}",f"{config['W1']},{config['W2']},{config['W3']},{config['W4']}","emg",f)
        accuracy_lda_forearm,_ = train_LDA(gesture_dfs,f"forearm_{f}",f"{config['F1']},{config['F2']},{config['F3']},{config['F4']}","emg",f)
        accuracy_svm_wrist,_ = train_svm(gesture_dfs,f"wrist_{f}",f"{config['W1']},{config['W2']},{config['W3']},{config['W4']}","emg",f)
        accuracy_svm_forearm,_ = train_svm(gesture_dfs,f"forearm_{f}",f"{config['F1']},{config['F2']},{config['F3']},{config['F4']}","emg",f)
        new_row = {
            "accuracy_lda_wrist": accuracy_lda_wrist,
            "accuracy_lda_forearm" : accuracy_lda_forearm ,
             "accuracy_svm_wrist": accuracy_svm_wrist,
             "accuracy_svm_forearm" : accuracy_svm_forearm
        }
        table6_df = pd.concat([table6_df, pd.DataFrame([new_row])], ignore_index=True)
    table6_df.to_csv(os.path.join(directory,"table6_df.csv"))
    
    #table7
    table7_df  = pd.DataFrame()
    for t in ["accel","gyro","mag","*","emg"]:
        accuracy_lda, _  =  train_LDA(gesture_dfs,f"table_7_lda_{t}","*",t,"*")
        accuracy_svm, _  =  train_LDA(gesture_dfs,f"table_7_svm_{t}","*",t,"*")
        new_row  = {
            "accuracy_lda" : accuracy_lda, 
            "accuracy_svm" : accuracy_svm 
        }
        table7_df = pd.concat([table7_df, pd.DataFrame([new_row])], ignore_index=True)
    table7_df.to_csv(os.path.join(directory,"table7_df.csv"))


    #table8 
    table8_df = pd.DataFrame()
    accuracy_anterior_wrist_lda = train_LDA(gesture_dfs,"table8_anterior_wrist_lda",f"{config["W1"]},{config["W2"]}","all","*")
    accuracy_posterior_wrist_lda = train_LDA(gesture_dfs,"table8_posterior_wrist_lda",f"{config["W3"]},{config["W4"]}","all","*")
    accuracy_anterior_forearm_lda = train_LDA(gesture_dfs,"table8_anterior_wrist_lda",f"{config["F1"]},{config["F2"]}","all","*")
    accuracy_posterior_forearm_lda = train_LDA(gesture_dfs,"table8_posterior_wrist_lda",f"{config["F3"]},{config["F4"]}","all","*")
    accuracy_wrist_lda = train_LDA(gesture_dfs,"table8_wrist_lda",f"{config["W1"]},{config["W2"]},{config["W3"]},{config["W4"]}","all","*")
    accuracy_forearm_lda = train_LDA(gesture_dfs,"table8_forearm_lda",f"{config["F1"]},{config["F2"]},{config["F3"]},{config["F4"]}","all","*")
    
    accuracy_anterior_wrist_svm = train_svm(gesture_dfs,"table8_anterior_wrist_svm",f"{config["W1"]},{config["W2"]}","all","*")
    accuracy_posterior_wrist_svm = train_svm(gesture_dfs,"table8_posterior_wrist_svm",f"{config["W3"]},{config["W4"]}","all","*")
    accuracy_anterior_forearm_svm = train_svm(gesture_dfs,"table8_anterior_wrist_svm",f"{config["F1"]},{config["F2"]}","all","*")
    accuracy_posterior_forearm_svm = train_svm(gesture_dfs,"table8_posterior_wrist_svm",f"{config["F3"]},{config["F4"]}","all","*")
    accuracy_wrist_svm = train_svm(gesture_dfs,"table8_wrist_svm",f"{config["W1"]},{config["W2"]},{config["W3"]},{config["W4"]}","all","*")
    accuracy_forearm_svm = train_svm(gesture_dfs,"table8_forearm_svm",f"{config["F1"]},{config["F2"]},{config["F3"]},{config["F4"]}","all","*")
    
    new_row = {"accuracy_anterior_wrist_lda": accuracy_anterior_wrist_lda, 
               "accuracy_posterior_wrist_lda" : accuracy_posterior_wrist_lda,
               "accuracy_anterior_forearm_lda" : accuracy_anterior_forearm_lda,
               "accuracy_posterior_forearm_lda" : accuracy_posterior_forearm_lda,
               "accuracy_wrist_lda": accuracy_wrist_lda,
               "accuracy_forearm_lda": accuracy_forearm_lda,

               "accuracy_anterior_wrist_svm": accuracy_anterior_wrist_svm, 
               "accuracy_posterior_wrist_svm" : accuracy_posterior_wrist_svm,
               "accuracy_anterior_forearm_svm" : accuracy_anterior_forearm_svm,
               "accuracy_posterior_forearm_svm" : accuracy_posterior_forearm_svm,
               "accuracy_wrist_svm": accuracy_wrist_svm,
               "accuracy_forearm_svm": accuracy_forearm_svm,
    }
    table8_df = pd.concat([table8_df, pd.DataFrame([new_row])], ignore_index=True)
    table8_df.to_csv(os.path.join(directory,"table8_df.csv"))



    # train_LDA(gesture_dfs)
    # train_svm(gesture_dfs)
    for i in range(len(gesture_dfs)):
        gesture_dfs[i].to_csv(os.path.join(directory,f"gesture_features{i}.csv"))

    print("done")