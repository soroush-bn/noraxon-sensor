import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import pandas as pd
from functools import wraps
from time import time
import argparse
import subprocess
import yaml
import numpy as np
import os

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")
if directory and not os.path.exists(directory):
    os.makedirs(directory)

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

@timing
def show_slides(experiment_name,first_name,last_name,reps,slides_images_path,gesture_duration,rest_duration):


    image_paths = [f"{slides_images_path}/{i}.jpg" for i in range(1, 17)]
    df = pd.DataFrame(image_paths, columns=['image_path'])

    sample_img = mpimg.imread(df.iloc[0]['image_path'])
    img_height, img_width, _ = sample_img.shape

    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100))  # Scale down by 100 for DPI adjustment
    plt.ion()
    fig.canvas.manager.full_screen_toggle()  # Open in full-screen mode
    names = ["Thumb Extension","index Extension","Middle Extension","Ring Extension",
             "Pinky Extension","Thumbs Up","Right Angle","Peace","OK","Horn","Hang Loose",
             "Power Grip","Hand Open","Wrist Extension","Wrist Flexion","Ulnar deviation","Radial Deviation"]
    show_phase1(ax)
    #start the script

    timestamped_labels = [] 
    for index, row in df.iterrows():
        img = mpimg.imread(row['image_path'])
        for rep in range(reps):        
            ax.clear()
            ax.imshow(img)
            
            ax.axis('off')  # Turn off axis
            image_name = names[index]
            timestamped_labels.append((time(), image_name))
            ax.text(0.5, 0.95, image_name, transform=ax.transAxes, 
                    ha='center', va='top', fontsize=20, fontweight='bold', color='white')
            plt.pause(gesture_duration)
            timestamped_labels.append((time(), "rest"))
            show_rest(ax,rest_duration)


    print("finished slides")
    plt.ioff()
    # plt.show()
    plt.close()
    timestamps_df = pd.DataFrame(timestamped_labels, columns=['time_stamp', 'label'])
    save_labels(timestamps_df)


def show_rest(ax,rest_duration):
    img = mpimg.imread(r"experiment1_images\rest.jpg")
        
    ax.clear()
    ax.imshow(img)
    ax.axis('off')  # Turn off axis
    ax.text(0.5, 0.95, "Rest", transform=ax.transAxes, 
                    ha='center', va='top', fontsize=20, fontweight='bold', color='white')

    plt.pause(rest_duration)

def show_phase1(ax):
    img = mpimg.imread(r"experiment1_images\phase1.jpg")
        
    ax.clear()
    ax.imshow(img)
    ax.axis('off')  # Turn off axis
    ax.text(0.5, 0.95, "Phase1", transform=ax.transAxes, 
                    ha='center', va='top', fontsize=20, fontweight='bold', color='white')

    plt.pause(config["phase1_wait"])
def save_labels(df): #populating labels
    if len(df)>200: 
        raise Exception("cannot extend the df multiple times!!!!")
    expanded_df = pd.DataFrame(columns=df.columns)
    for i in range(len(df) - 1):
        start_time = df.loc[i, 'time_stamp']
        end_time = df.loc[i + 1, 'time_stamp']
        label = df.loc[i, 'label']
        
        time_diff = end_time - start_time
        num_rows = int(time_diff * config["emg_frequency"])  # 2000 rows per second
        
        new_time_stamps = np.linspace(start_time, end_time, num=num_rows, endpoint=False)
        new_rows = pd.DataFrame({
            'time_stamp': new_time_stamps,
            'label': [label] * num_rows
        })
        
        expanded_df = pd.concat([expanded_df, new_rows], ignore_index=True)

    expanded_df = pd.concat([expanded_df, df.iloc[[-1]]], ignore_index=True)
    expanded_df.reset_index(drop=True, inplace=True)
    expanded_df.to_csv(f'{directory}labels.csv')
    print("label df saved")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="no desc.")
    parser.add_argument("--first_name", type=str, default=config["first_name"])
    parser.add_argument("--last_name", type=str, default=config["last_name"])
    parser.add_argument("--experiment_name", type=str, default=config["experiment_name"])
    parser.add_argument("--slides_images_path", type=str, default=config["slides_images_path"])
    
    parser.add_argument("--reps", type=str, default=config["number_of_gesture_repetition"])
    parser.add_argument("--gesture_duration", type=int, default=config["gesture_duration"])
    parser.add_argument("--rest_duration", type=int, default=config["rest_duration"])
    
    args = parser.parse_args()
    print("\n starting experiment with this config: \n")
    print(args)
    print()

    command = ["python", "recorder.py"]
    subprocess.Popen(command)
    show_slides(args.experiment_name,args.first_name,args.last_name,args.reps,args.slides_images_path,args.gesture_duration,args.rest_duration)
