# import subprocess

# command = ["python", "recorder.py", "--name", "soroush.avi"]


# subprocess.run(command)
#todo no need this, might be deleted later.


import pyautogui
import subprocess
import time

# Target position for the button click (replace these with actual coordinates)
target_x, target_y = 500, 300  # Example coordinates8
tolerance = 500  # Allowable range around the target coordinates for flexibility

# Path to the script you want to run
script_path = "C:/path/to/your/script.bat"

def is_mouse_on_target():
    # Get current mouse position
    mouse_x, mouse_y = pyautogui.position()
    # Check if the mouse is within the target coordinates (with tolerance)
    return (target_x - tolerance <= mouse_x <= target_x + tolerance) and \
           (target_y - tolerance <= mouse_y <= target_y + tolerance)

def main():
    print("Monitoring mouse clicks...")
    while True:
        # Check if the left mouse button is clicked
        if is_mouse_on_target() and pyautogui.mouseDown(button="left"):
            print("Target button clicked!")
            # Run the external script
            # subprocess.run([script_path], shell=True)
            # Add a delay to prevent multiple triggers
            time.sleep(1)
        # Small delay to avoid high CPU usage
        time.sleep(0.1)

if __name__ == "__main__":
    main()
