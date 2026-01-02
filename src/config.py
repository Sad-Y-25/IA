import os
import xml.etree.ElementTree as ET

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Point to the complex map in data/networks
SUMO_CFG_PATH = os.path.join(DATA_DIR, 'networks', 'RL.sumocfg')
SUMO_GUI_BINARY = "sumo-gui"
STEP_LENGTH = "0.5"

# --- VISION ---
# Save in the root project folder
IMAGE_CAPTURE_PATH = os.path.join(PROJECT_ROOT, "sumo_capture.png")
DEBUG_FOLDER = os.path.join(PROJECT_ROOT, "debug_captures")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, 'models', 'yolov8n-seg.pt')

# --- SIMULATION SETTINGS ---
TOTAL_STEPS = 400  
STANDARD_MIN_GREEN = 10
ACTIONS = [0, 1] 

# --- Q-LEARNING ---
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# --- DYNAMIC CONFIG LOADING ---
def load_detector_groups():
    """Parses RL.add.xml to group detector IDs by node (e.g., Node2 has multiple)."""
    add_xml_path = os.path.join(DATA_DIR, 'networks', 'RL.add.xml')
    groups = {}
    if os.path.exists(add_xml_path):
        try:
            tree = ET.parse(add_xml_path)
            root = tree.getroot()
            for detector in root.findall('laneAreaDetector'):
                det_id = detector.get('id')
                node_id = det_id.split('_')[0]  # e.g., 'Node2' from 'Node2_1_EB_0'
                if node_id not in groups:
                    groups[node_id] = []
                groups[node_id].append(det_id)
            print(f"✅ Loaded {sum(len(g) for g in groups.values())} detectors grouped into {len(groups)} nodes.")
        except Exception as e:
            print(f"⚠️ Error loading detector groups: {e}")
    else:
        print(f"⚠️ Could not find {add_xml_path}")
    return groups

DETECTOR_GROUPS = load_detector_groups()
TRAFFIC_LIGHT_IDS = ['Node1', 'Node2', 'Node3', 'Node5', 'Node7']