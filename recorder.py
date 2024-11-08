import pyrealsense2 as rs
import numpy as np
import cv2
import os

bag_file_path = "path_to_your_file.bag"  
output_dir = "output_images"  

os.makedirs(output_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file_path)

pipeline.start(config)

try:
    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        
        image_filename = os.path.join(output_dir, f"image_{frame_count:04d}.png")
        cv2.imwrite(image_filename, color_image)
        frame_count += 1

        cv2.imshow('Color Frame', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
