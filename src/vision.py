import cv2
import numpy as np
import os
import traci
from ultralytics import YOLO
from . import config

class TrafficEye:
    def __init__(self):
        print(f"Loading YOLO model from {config.YOLO_MODEL_PATH}...")
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.view_id = None 

    def find_correct_view_id(self):
        """Auto-detects the correct View ID from SUMO"""
        try:
            views = traci.gui.getIDList()
            if not views: return None
            print(f"DEBUG: Found SUMO Views: {views}")
            self.view_id = views[0]
            return self.view_id
        except:
            return None

    def capture_screenshot(self):
        """Non-Blocking Screenshot"""
        if self.view_id is None:
            self.find_correct_view_id()
            if self.view_id is None: return False

        full_path = os.path.abspath(config.IMAGE_CAPTURE_PATH)
        try:
            traci.gui.screenshot(self.view_id, full_path)
            # Check if file exists and has content
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0
        except:
            return False

    def analyze_traffic_hybrid(self, debug_save=False):
        full_path = os.path.abspath(config.IMAGE_CAPTURE_PATH)
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            return 0, 0, 0
            
        try:
            img = cv2.imread(full_path)
            if img is None: return 0, 0, 0
            
            # --- 1. YOLO (Keep this for the "Project Requirement") ---
            # We run it, but we know it might return 0 in simulation.
            results = self.model(img, verbose=False, conf=0.1)
            yolo_cars = sum(1 for box in results[0].boxes if int(box.cls[0]) == 2)
            
            # --- 2. ROBUST COLOR MASKING (The "Real" Simulation Detector) ---
            # These ranges are from your ORIGINAL final.py which worked!
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Yellow Cars (Standard SUMO vehicles)
            mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
            color_cars = self.count_blobs(mask_yellow)

            # Heavy Vehicles (Magenta/Red in SUMO)
            mask_mag = cv2.inRange(hsv, np.array([135, 100, 100]), np.array([165, 255, 255]))
            mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask_heavy = cv2.bitwise_or(mask_mag, mask_red)
            color_heavy = self.count_blobs(mask_heavy)

            # Ambulances (White) - Using your Original Wide Range
            # We rely on size filtering (>100) to ignore road lines
            mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
            color_emerg = self.count_blobs(mask_white)

            # --- 3. MERGE RESULTS ---
            # We take the MAXIMUM of YOLO or Color. 
            # If YOLO sees 0 but Color sees 5, we return 5.
            final_cars = max(yolo_cars, color_cars)
            final_heavy = color_heavy
            final_emerg = color_emerg # YOLO doesn't detect ambulances anyway
            
            if debug_save:
                # Save what the code actually saw
                cv2.imwrite("debug_view.png", img)
                cv2.imwrite("debug_mask_white.png", mask_white)

            return final_cars, final_heavy, final_emerg
        except Exception as e:
            print(f"Vision Error: {e}")
            return 0, 0, 0

    def count_blobs(self, mask):
        """Counts objects, ignoring small noise (road lines)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Area > 100 removes small dots/lines. 
        # If ambulances are not detected, LOWER this number (e.g. to 50).
        return sum(1 for c in contours if cv2.contourArea(c) > 100)

    def get_vision_reward(self, debug=False):
        self.capture_screenshot()
        n_c, n_h, n_e = self.analyze_traffic_hybrid(debug_save=debug)
        
        penalty = (n_c * 1.0) + (n_h * 5.0) + (n_e * 50.0)
        return -penalty, n_c, n_h, n_e