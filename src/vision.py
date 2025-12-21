import cv2
import numpy as np
import time
import traci
from . import config

def capture_screenshot(view_id="View #0"):
    """Ask SUMO to save a screenshot."""
    try:
        traci.gui.screenshot(view_id, config.IMAGE_CAPTURE_PATH)
        time.sleep(0.05) # Give file system time to write
        return True
    except Exception as e:
        print(f"Screenshot failed: {e}")
        return False

def analyze_traffic_image(debug_save=False):
    """
    Reads the screenshot and detects Cars, Heavy Vehicles, and Emergencies.
    Returns: (count_cars, count_heavy, count_emergency)
    """
    img = cv2.imread(config.IMAGE_CAPTURE_PATH)
    if img is None:
        return 0, 0, 0

    if debug_save:
        cv2.imwrite("debug_original.png", img)

    # Preprocessing
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- MASKS (From your final.py) ---
    # Cars (Yellow)
    mask_cars = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

    # Heavy (Magenta/Red)
    mask_mag = cv2.inRange(hsv, np.array([135, 100, 100]), np.array([165, 255, 255]))
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    mask_heavy = cv2.bitwise_or(mask_mag, mask_red)

    # Emergency (White - Low saturation, High Value)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_emerg = cv2.inRange(hsv, lower_white, upper_white)

    # Cleanup noise
    kernel = np.ones((5, 5), np.uint8)
    mask_emerg = cv2.dilate(cv2.erode(mask_emerg, kernel, iterations=1), kernel, iterations=2)
    # (Repeat for others if needed)

    # Counting
    def count_contours(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if cv2.contourArea(c) > 100)

    n_cars = count_contours(mask_cars)
    n_heavy = count_contours(mask_heavy)
    n_emerg = count_contours(mask_emerg)

    return n_cars, n_heavy, n_emerg

def get_vision_reward(debug=False):
    """Calculates reward based on visual congestion."""
    if not capture_screenshot():
        return 0, 0, 0, 0

    n_c, n_h, n_e = analyze_traffic_image(debug_save=debug)
    
    # Penalty Formula
    penalty = (n_c * 1.0) + (n_h * 10.0) + (n_e * 100.0)
    return -penalty, n_c, n_h, n_e