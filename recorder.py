# this file should give you record.avi file in data/first_lastname_experimentname/ folder
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import pandas as pd
import yaml 
import os

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")

# Create directory if it doesn't exist
if directory and not os.path.exists(directory):
    os.makedirs(directory)

def record():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Use os.path.join() to create paths
    color_path = os.path.join(directory, 'record.avi')
    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), 1)

    data = []
    pipeline.start(config)
    print("start recording")
    frame_index = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            colorwriter.write(color_image)
            
            data.append([time.time(), frame_index])
            frame_index += 1

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        colorwriter.release()
        print("stopped recording")
        pipeline.stop()

        # Use os.path.join() to save the CSV
        csv_path = os.path.join(directory, 'frames.csv')
        df = pd.DataFrame(data, columns=["time_stamp", "frame"])
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="no desc.")
    args = parser.parse_args()
    record()
