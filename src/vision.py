import cv2
import numpy as np
import os
import traci
import time
import tempfile
from ultralytics import YOLO
from . import config

class TrafficEye:
    def __init__(self):
        print(f"Loading YOLO model from {config.YOLO_MODEL_PATH}...")
        try:
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Vision system will use color-based detection only")
            self.model = None
            
        self.view_id = None 
        self.debug_folder = "debug_screenshots"
        self.debug_counter = 0
        self.max_debug_files = 25  # Keep only last 25 debug files
        
        # Create folders
        for folder in [self.debug_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            
    def cleanup_old_debug_files(self):
        """Keep only the most recent debug files."""
        try:
            files = [f for f in os.listdir(self.debug_folder) if f.endswith(('.png', '.jpg', '.txt'))]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.debug_folder, x)))
            
            # Remove oldest files if we have more than max_debug_files
            if len(files) > self.max_debug_files:
                for file in files[:-self.max_debug_files]:
                    os.remove(os.path.join(self.debug_folder, file))
        except:
            pass
    
    def find_view_id(self):
        """Find the SUMO GUI view ID."""
        try:
            if self.view_id is None:
                views = traci.gui.getIDList()
                if views:
                    self.view_id = views[0]
                    print(f"Found SUMO view: {self.view_id}")
                    
                    # Removed zooming and offset to avoid annoying initial zoom
            return self.view_id
        except Exception as e:
            print(f"Error finding view ID: {e}")
            return None

    def capture_screenshot(self):
    # Capture a screenshot from SUMO.
        if self.view_id is None:
            self.find_view_id()
            if self.view_id is None:
                return None
    
    # Use temp file in debug_folder
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', dir=self.debug_folder, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            traci.gui.screenshot(self.view_id, temp_path)
            time.sleep(0.5)  # Increased delay
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                return temp_path
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None
        except Exception as e:
            print(f"Screenshot error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def detect_vehicles(self, img, save_debug=False, step=0):
        """Detect vehicles using color filtering (with pedestrian support)."""
        if img is None:
            return 0, 0, 0, 0
            
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = img.shape[:2]
            
            kernel = np.ones((3,3), np.uint8)
            
            # ===== COLOR FILTERS =====
            # Cars - Yellow (default vehicles, buses, etc.)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask_cars = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_cars = cv2.dilate(mask_cars, kernel, iterations=2)
            
            # Ambulances/Emergency - Red
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_ambulances = cv2.bitwise_or(mask_red1, mask_red2)
            mask_ambulances = cv2.dilate(mask_ambulances, kernel, iterations=2)
            
            # Big Trucks - White/bright (heavy vehicles)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_trucks = cv2.inRange(hsv, lower_white, upper_white)
            mask_trucks = cv2.dilate(mask_trucks, kernel, iterations=1)
            
            # Pedestrians - Dark gray (default person color in SUMO)
            lower_gray = np.array([0, 0, 0])
            upper_gray = np.array([180, 50, 150])  # Low sat, medium value
            mask_peds = cv2.inRange(hsv, lower_gray, upper_gray)
            mask_peds = cv2.dilate(mask_peds, kernel, iterations=1)
            
            # Count contours (filter by size to avoid noise)
            def count_contours(mask, min_area=20, max_area=500):
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                return sum(1 for cnt in contours if min_area < cv2.contourArea(cnt) < max_area)
            
            peds = count_contours(mask_peds, min_area=2, max_area=20)  # Small for peds
            cars = count_contours(mask_cars)
            ambulances = count_contours(mask_ambulances)
            trucks = count_contours(mask_trucks)
            
            if save_debug:
                debug_id = str(time.time()).replace('.', '')
                vis_img = img.copy()
                
                # Draw contours
                contours_cars = cv2.findContours(mask_cars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours_ambulances = cv2.findContours(mask_ambulances, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours_trucks = cv2.findContours(mask_trucks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours_peds = cv2.findContours(mask_peds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                
                cv2.drawContours(vis_img, contours_cars, -1, (0, 255, 255), 2)  # Yellow for cars
                cv2.drawContours(vis_img, contours_ambulances, -1, (0, 0, 255), 2)  # Red for ambulances
                cv2.drawContours(vis_img, contours_trucks, -1, (255, 255, 255), 2)  # White for trucks
                cv2.drawContours(vis_img, contours_peds, -1, (255, 0, 255), 1)  # Magenta for peds
                
                # Add text
                cv2.putText(vis_img, f"Peds: {peds}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(vis_img, f"Cars: {cars}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis_img, f"Ambulances: {ambulances}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(vis_img, f"Trucks: {trucks}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(vis_img, f"Step: {step}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save debug images
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_original.png"), img)
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_detection.png"), vis_img)
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_mask_peds.png"), mask_peds)
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_mask_cars.png"), mask_cars)
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_mask_ambulances.png"), mask_ambulances)
                cv2.imwrite(os.path.join(self.debug_folder, f"{debug_id}_mask_trucks.png"), mask_trucks)
                
                # Create summary
                summary_path = os.path.join(self.debug_folder, f"{debug_id}_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"Debug ID: {debug_id}\n")
                    f.write(f"Step: {step}\n")
                    f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
                    f.write(f"\nDetection Results:\n")
                    f.write(f"  Peds: {peds}\n")
                    f.write(f"  Cars: {cars}\n")
                    f.write(f"  Ambulances: {ambulances}\n")
                    f.write(f"  Trucks: {trucks}\n")
                
                print(f"Saved debug images #{debug_id}")
                
                # Clean up old files
                self.cleanup_old_debug_files()
            
            return peds, cars, ambulances, trucks
            
        except Exception as e:
            print(f"Detection error: {e}")
            return 0, 0, 0, 0

    def analyze_traffic(self, step=0, debug=False):
        """Main traffic analysis function."""
        # Capture screenshot
        screenshot_path = self.capture_screenshot()
        if screenshot_path is None:
            if debug:
                print("Failed to capture screenshot")
            return 0, 0, 0, 0
        
        # Load image
        img = cv2.imread(screenshot_path)
        
        # Clean up temp file
        try:
            os.remove(screenshot_path)
        except:
            pass
        
        if img is None:
            return 0, 0, 0, 0
        
        # Detect vehicles
        return self.detect_vehicles(img, save_debug=debug, step=step)

    def get_vision_reward(self, step=0, debug=False):
    # """Calculate reward based on vision analysis."""
        if not debug:
            return 0, 0, 0, 0, 0  # Skip vision for non-detailed steps
        
        print(f"\n--- Step {step}: Vision Analysis ---")
        
        peds, cars, ambulances, trucks = self.analyze_traffic(step=step, debug=debug)
        
        # Calculate penalty
        penalty = (peds * 1.5) + (cars * 1.0) + (trucks * 3.0) + (ambulances * 50.0)
        
        print(f"Detected - Peds: {peds}, Cars: {cars}, Ambulances: {ambulances}, Trucks: {trucks}")
        print(f"Penalty: {penalty:.1f} (Peds×1.5 + Cars×1 + Trucks×3 + Ambulances×50)")
        print(f"Vision reward: {-penalty:.1f}")
        
        return -penalty, peds, cars, ambulances, trucks