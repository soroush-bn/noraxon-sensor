import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import pandas as pd
from functools import wraps
from time import time
import argparse


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
def show_slides(name,reps):
    image_paths = [f"experiment1_images/{i}.jpg" for i in range(1, 17)]
    df = pd.DataFrame(image_paths, columns=['image_path'])

    sample_img = mpimg.imread(df.iloc[0]['image_path'])
    img_height, img_width, _ = sample_img.shape

    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100))  # Scale down by 100 for DPI adjustment
    plt.ion()
    fig.canvas.manager.full_screen_toggle()  # Open in full-screen mode
    names = ["Thumb Extension","index Extension","Middle Extension","Ring Extension",
             "Pinky Extension","Thumbs Up","Right Angle","Peace","OK","Horn","Hang Loose",
             "Power Grip","Hand Open","Wrist Extension","Wrist Flexion","Ulnar deviation","Radial Deviation"]
    for index, row in df.iterrows():
        img = mpimg.imread(row['image_path'])
        for rep in range(reps):        
            ax.clear()
            ax.imshow(img)
            
            ax.axis('off')  # Turn off axis
            image_name = names[index]
            ax.text(0.5, 0.95, image_name, transform=ax.transAxes, 
                    ha='center', va='top', fontsize=20, fontweight='bold', color='white')
            plt.pause(2)
            show_rest(ax)


    print("finished slides")
    plt.ioff()
    # plt.show()
    plt.close()



def show_rest(ax):
    img = mpimg.imread(r"experiment1_images\rest.jpg")
        
    ax.clear()
    ax.imshow(img)
    ax.axis('off')  # Turn off axis
    ax.text(0.5, 0.95, "Rest", transform=ax.transAxes, 
                    ha='center', va='top', fontsize=20, fontweight='bold', color='white')

    plt.pause(2)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="no desc.")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--reps", type=int, required=True)
    
    args = parser.parse_args()
    show_slides(args.name,args.reps)