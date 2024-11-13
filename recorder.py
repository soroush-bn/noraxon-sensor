import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import pandas as pd

# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()

#working one 
def record(name="saved.avi"):
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    color_path = name
    # depth_path = 'V00P00A00C00_depth.avi'
    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
    # depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
    data = []
    pipeline.start(config)
    print("start recording")
    frame_index=0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if  not color_frame: #not depth_frame or
                continue
            
            #convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            colorwriter.write(color_image)
            # depthwriter.write(depth_colormap)
            
            # cv2.imshow('Stream', depth_colormap)
            data.append([time.time(), frame_index])
            frame_index+=1
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        colorwriter.release()
        # depthwriter.release()
        print("stoped recording")
        pipeline.stop()
        df = pd.DataFrame(data, columns=["time_stamp", "frame"])
        df.to_csv(f"{name}_frame_time.csv", index=False)
        print(f"Data saved to {name}_frame_time.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="no desc.")
    parser.add_argument("--name", type=str, required=True)
    
    args = parser.parse_args()
    record(args.name)