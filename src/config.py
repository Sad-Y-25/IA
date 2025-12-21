import os

# --- PATHS ---
# Base paths relative to the main script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# SUMO Config
SUMO_CFG_PATH = os.path.join(DATA_DIR, 'networks', 'RL.sumocfg')
SUMO_GUI_BINARY = "sumo-gui"  # Or full path if needed
STEP_LENGTH = "0.5"

# Vision
IMAGE_CAPTURE_PATH = "sumo_capture.png"
YOLO_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'yolov8n-seg.pt')

# --- SIMULATION SETTINGS ---
TOTAL_STEPS = 200
STANDARD_MIN_GREEN = 10
ACTIONS = [0, 1]  # 0: Keep/Switch North, 1: Keep/Switch East (Depends on phase)

# --- Q-LEARNING HYPERPARAMETERS ---
ALPHA = 0.1     # Learning Rate
GAMMA = 0.9     # Discount Factor
EPSILON = 0.1   # Exploration Rate

# --- DETECTORS (Sensors in the road) ---
# These IDs match your .add.xml file
DETECTOR_IDS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
]