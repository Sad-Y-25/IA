import os

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Map Config
SUMO_CFG_PATH = os.path.join(DATA_DIR, 'networks', 'RL.sumocfg')
SUMO_GUI_BINARY = "sumo-gui"
STEP_LENGTH = "0.5"

# --- VISION ---
# FIX: Use a simple filename. No absolute paths with backslashes.
IMAGE_CAPTURE_PATH = "sumo_capture.png"
YOLO_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'yolov8n-seg.pt')

# --- SIMULATION SETTINGS ---
TOTAL_STEPS = 200
STANDARD_MIN_GREEN = 10
ACTIONS = [0, 1] 

# --- Q-LEARNING ---
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# --- DETECTORS ---
DETECTOR_IDS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
]