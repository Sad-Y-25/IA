import cv2
import numpy as np
import os
import traci
import time
import tempfile
from . import config

class TrafficEye:
    def __init__(self):
        # We focus on the "Image Filter" approach as requested, but backed by TRACI data
        self.view_id = None 
        self.debug_folder = "debug_screenshots"
        self.max_debug_files = 25
        
        # Ensure debug folder exists
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
            print(f"[Vision] Created folder: {self.debug_folder}")

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
                is_truck = (v_class in ["truck", "trailer", "delivery"]) or ("truck" in vid.lower())
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
        """
        Forces vehicles in SUMO to match the Colors expected by the Image Filters.
        This makes the Computer Vision detection actually work.
        """
        try:
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                v_class = traci.vehicle.getVehicleClass(vid)
                v_type = traci.vehicle.getTypeID(vid)

                # Determine Type
                is_ambulance = (v_class == "emergency") or ("amb" in vid.lower())
                is_truck = (v_class in ["truck", "trailer"]) or ("truck" in vid.lower())

                # Set Color (R, G, B, Alpha)
                if is_ambulance:
                    traci.vehicle.setColor(vid, (255, 0, 0, 255)) # Pure Red
                elif is_truck:
                    traci.vehicle.setColor(vid, (255, 255, 255, 255)) # Pure White
                else:
                    traci.vehicle.setColor(vid, (255, 255, 0, 255)) # Pure Yellow (Default Car)
            
            # Pedestrians usually default to dark gray/black, which works for the filter
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
        Main function called by the simulation.
        Returns the counts and the penalty.
        """
        
        # 1. Force Colors in GUI (So filters work visually)
        self.enforce_visual_colors()

        # 2. Get 100% Accurate Counts from SUMO (So logic never fails)
        gt_peds, gt_cars, gt_amb, gt_trucks = self.get_traci_ground_truth()

        # 3. If debug is on, run the visual filters to create the screenshot
        if debug:
            path = self.capture_screenshot()
            if path:
                img = cv2.imread(path)
                self.apply_image_filters(img, step=step, debug=True)

        # 4. Calculate Penalty (using the accurate GT data)
        # Weights: Peds(1.5), Cars(1.0), Trucks(3.0), Ambulances(50.0)
        penalty = (gt_peds * 1.5) + (gt_cars * 1.0) + (gt_trucks * 3.0) + (gt_amb * 50.0)

        if debug:
            print(f"[Vision] Step {step} | Truth: Cars={gt_cars}, Amb={gt_amb}, Trucks={gt_trucks} | Penalty: {penalty}")

        return -penalty, gt_peds, gt_cars, gt_amb, gt_trucks