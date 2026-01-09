import cv2
import numpy as np
import os
import traci
import time
import tempfile
from . import config
from ultralytics import YOLO 
import torch

class TrafficEye:
    def __init__(self):
        # We focus on the "Image Filter" approach as requested, but backed by TRACI data
        self.view_id = None 
        self.debug_folder = "debug_screenshots"
        self.max_debug_files = 25
        
        # --- YOLO ---
        self.yolo_model = None
        try:
            # Pointing to the specific file you uploaded
            model_path = os.path.join(config.DATA_DIR, 'models', 'yolov8n-seg.pt') 
            self.yolo_model = YOLO(model_path)
            print(f"✅ Loaded YOLOv8 Segmentation model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
        # ----------------------

        # Ensure debug folder exists
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
            print(f"[Vision] Created folder: {self.debug_folder}")
       
    def analyze_with_yolo(self, img):
        """
        Runs YOLOv8 Segmentation on top of the image with specific coloring.
        Colors (BGR): Ambulance=White, Truck=Blue, Bus=Yellow.
        """
        if self.yolo_model is None or img is None:
            return img, 0, 0, 0, 0

        # Run inference
        results = self.yolo_model(img, verbose=False, conf=0.4)
        result = results[0]
        
        y_peds = 0
        y_cars = 0
        y_amb = 0
        y_trucks = 0
        
        # Check if boxes exist
        if result.boxes:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
                
                # Crop for color check
                roi = img[y1:y2, x1:x2]
                is_ambulance = False
                
                # YOLO Classes: 0=Person, 2=Car, 5=Bus, 7=Truck
                if cls_id in [2, 5, 7]: 
                    # --- AMBULANCE LOGIC (Red Check) ---
                    if roi.size > 0:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # Red Mask (Lower and Upper ranges)
                        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                        red_mask = cv2.bitwise_or(mask1, mask2)
                        
                        # If >15% red pixels, assume Ambulance
                        if (cv2.countNonZero(red_mask) / roi.size) > 0.15:
                            is_ambulance = True
                            y_amb += 1
                            # AMBULANCE -> WHITE [255, 255, 255]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                            cv2.putText(img, "AMBULANCE", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    if not is_ambulance:
                        if cls_id == 7: # Truck
                            y_trucks += 1
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue Box
                            cv2.putText(img, "TRUCK", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        elif cls_id == 5: # Bus
                            y_trucks += 1 
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow Box
                            cv2.putText(img, "BUS", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        else: # Car
                            y_cars += 1
                            # Draw Green Box for Cars
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                            cv2.putText(img, "CAR", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif cls_id == 0: # Person
                    y_peds += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

        return img, y_peds, y_cars, y_amb, y_trucks     

    def cleanup_old_debug_files(self):
        """Keep only the most recent debug files to save space."""
        try:
            files = [f for f in os.listdir(self.debug_folder) if f.endswith(('.png', '.jpg', '.txt'))]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.debug_folder, x)))
            if len(files) > self.max_debug_files:
                for file in files[:-self.max_debug_files]:
                    os.remove(os.path.join(self.debug_folder, file))
        except:
            pass

    def get_traci_ground_truth(self):
        """
        DIRECTLY detects vehicles using SUMO IDs and Configuration.
        This ensures values are NEVER 0 if vehicles exist.
        """
        peds_count = 0
        cars_count = 0
        amb_count = 0
        truck_count = 0

        # 1. Count Pedestrians
        try:
            peds_count = len(traci.person.getIDList())
        except:
            peds_count = 0

        # 2. Count Vehicles by Class/Type
        try:
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                v_class = traci.vehicle.getVehicleClass(vid) # e.g., 'passenger', 'truck', 'emergency'
                v_type = traci.vehicle.getTypeID(vid)
                
                # Logic to classify vehicles
                is_ambulance = (v_class == "emergency") or ("amb" in vid.lower()) or ("emergency" in v_type.lower())
                is_truck = (v_class in ["truck", "trailer", "delivery","bus"]) or ("truck" in vid.lower())
                is_car = (v_class in ["passenger", "taxi", "hov"]) or (not is_ambulance and not is_truck)

                if is_ambulance:
                    amb_count += 1
                elif is_truck:
                    truck_count += 1
                elif is_car:
                    cars_count += 1
                    
        except Exception as e:
            print(f"[Vision Error] Could not retrieve TRACI data: {e}")

        return peds_count, cars_count, amb_count, truck_count

    def enforce_visual_colors(self):
        """Forces colors: Ambulance=Red, Truck=White, Bus=Yellow, Car=Green."""
        try:
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                v_class = traci.vehicle.getVehicleClass(vid)
                
                # Precise checks
                is_ambulance = (v_class == "emergency") or ("amb" in vid.lower())
                is_truck = (v_class in ["truck", "trailer", "delivery"])
                is_bus = (v_class == "bus") or ("bus" in vid.lower())
                
                if is_ambulance:
                    traci.vehicle.setColor(vid, (255, 0, 0, 255))   # Red
                elif is_truck:
                    traci.vehicle.setColor(vid, (255, 255, 255, 255)) # White
                elif is_bus:
                     traci.vehicle.setColor(vid, (255, 255, 0, 255))  # Yellow
                else:
                    traci.vehicle.setColor(vid, (0, 255, 0, 255))   # Green
        except:
            pass

    def capture_screenshot(self):
        """Captures the SUMO GUI."""
        if self.view_id is None:
            try:
                self.view_id = traci.gui.getIDList()[0]
                # Optional: Zoom out to see everything
                traci.gui.setZoom(self.view_id, 100) 
            except:
                return None
        
        temp_path = os.path.join(self.debug_folder, "temp_vision.png")
        try:
            traci.gui.screenshot(self.view_id, temp_path)
            # Small wait to ensure file write finishes
            time.sleep(0.1) 
            if os.path.exists(temp_path):
                return temp_path
        except:
            pass
        return None

    def apply_image_filters(self, img, step=0, debug=False):
        """
        Old-school Computer Vision: Color Filtering.
        Used for visualization and debug generation.
        """
        if img is None:
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- DEFINING COLOR MASKS (The "Filters") ---
        # 1. Yellow Cars
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_cars = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. Red Ambulances (Range 1 and Range 2 for Red wrapping)
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_amb = cv2.bitwise_or(mask_red1, mask_red2)

        # 3. White Trucks
        # High brightness, low saturation
        mask_trucks = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

        # 4. Pedestrians (Dark colors)
        mask_peds = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        if debug:
            # Draw contours on debug image
            vis_img = img.copy()
            
            def draw_cnt(mask, color, label):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) > 50: # Filter noise
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
                        count += 1
                return count

            c_cars = draw_cnt(mask_cars, (0, 255, 255), "Car")
            c_amb = draw_cnt(mask_amb, (0, 0, 255), "Amb")
            c_truck = draw_cnt(mask_trucks, (255, 255, 255), "Truck")
            c_peds = draw_cnt(mask_peds, (255, 0, 255), "Ped")

            # Save the annotated image
            debug_path = os.path.join(self.debug_folder, f"step_{step}_vision_filter.png")
            cv2.putText(vis_img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Visual Counts -> Cars:{c_cars} Amb:{c_amb} Truck:{c_truck}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(debug_path, vis_img)
            self.cleanup_old_debug_files()

    def get_vision_reward(self, step=0, debug=False):
        """
        Uses TRACI for logic counts (100% accurate).
        Uses YOLO+OpenCV for Debug Screenshots (Showoff).
        """
        # 1. Set Colors
        self.enforce_visual_colors()
        
        # 2. Get LOGIC DATA (Source of Truth)
        gt_peds, gt_cars, gt_amb, gt_trucks = self.get_traci_ground_truth()

        # 3. Create SCREENSHOT (Only if debug is ON)
        if debug:
            path = self.capture_screenshot()
            if path:
                img = cv2.imread(path)
                
                # Apply the Visual Showoff (OpenCV + YOLO)
                img_final = self.create_debug_visuals(img)
                
                # Save the file
                save_path = os.path.join(self.debug_folder, f"step_{step}_combined.jpg")
                cv2.imwrite(save_path, img_final)
                print(f"📸 Debug Image Saved: {save_path}")

        # 4. Return TRACI counts for the Agent
        penalty = (gt_peds * 1.5) + (gt_cars * 1.0) + (gt_trucks * 3.0) + (gt_amb * 50.0)
        return -penalty, gt_peds, gt_cars, gt_amb, gt_trucks
    
    def create_debug_visuals(self, img):
        """
        Applies OpenCV filters (Green Cars) AND YOLO overlays (Labels) for the screenshot.
        """
        if img is None: return img
        
        # --- 1. OPENCV SHOWOFF (Filter Green Cars) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Green mask for the cars we just painted
        mask_cars = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask_cars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw a thin Green box to show "OpenCV detected this color"
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # --- 2. YOLO SHOWOFF (Object Detection) ---
        if self.yolo_model:
            results = self.yolo_model(img, verbose=False, conf=0.3)
            if results[0].boxes:
                for box in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
                    
                    # Determine Label & Color based on YOLO Class
                    label = ""
                    color = (200, 200, 200) # Default Gray
                    
                    if cls_id == 2: # Car
                        label = "YOLO: Car"
                        color = (0, 255, 0)
                    elif cls_id == 5: # Bus
                        label = "YOLO: Bus"
                        color = (0, 255, 255)
                    elif cls_id == 7: # Truck
                        label = "YOLO: Truck"
                        color = (255, 0, 0) # Blue (BGR)
                    
                    # Hybrid Check for Ambulance (Red Color inside YOLO box)
                    roi = img[y1:y2, x1:x2]
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # Red Mask Range
                        mask_red1 = cv2.inRange(hsv_roi, np.array([0, 70, 50]), np.array([10, 255, 255]))
                        mask_red2 = cv2.inRange(hsv_roi, np.array([170, 70, 50]), np.array([180, 255, 255]))
                        red_ratio = cv2.countNonZero(cv2.bitwise_or(mask_red1, mask_red2)) / roi.size
                        
                        if red_ratio > 0.15:
                            label = "AMBULANCE"
                            color = (255, 255, 255) # White
                            # Draw thicker box for Ambulance
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # Draw the YOLO box/label if we found one
                    if label:
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        if label != "AMBULANCE": # Avoid drawing twice for ambulance
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img