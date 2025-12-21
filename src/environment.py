import os
import sys
import traci
import sumolib
from . import config

def check_sumo_installation():
    """Ensures SUMO_HOME is set and tools are in path."""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        sys.exit("Error: Please set the environment variable 'SUMO_HOME'.")

def start_simulation(use_gui=True):
    """Starts the SUMO simulation."""
    check_sumo_installation()
    
    sumo_binary = config.SUMO_GUI_BINARY if use_gui else "sumo"
    
    cmd = [
        sumo_binary,
        '-c', config.SUMO_CFG_PATH,
        '--step-length', config.STEP_LENGTH,
        '--lateral-resolution', '0',
        '--start'  # Auto start
    ]
    
    # Get a free port to avoid conflicts
    port = sumolib.miscutils.getFreeSocketPort()
    traci.start(cmd, port=port)
    
    # Set view to "real world" schema
    traci.gui.setSchema("View #0", "real world")

def close_simulation():
    traci.close()

def get_queue_lengths():
    """Returns a list of vehicle counts from the induction loops."""
    return [traci.lanearea.getLastStepVehicleNumber(d) for d in config.DETECTOR_IDS]

def set_traffic_light_phase(node_id, action):
    """
    Simulates switching phases. 
    Note: A simple implementation assuming 4 phases (0,1,2,3).
    Action 1 implies 'Switch' logic usually.
    """
    current_phase = traci.trafficlight.getPhase(node_id)
    # Simple logic: If action is 1, advance phase. 
    # Real logic depends on your specific traffic light program in net.xml
    if action == 1:
        new_phase = (current_phase + 1) % 4
        traci.trafficlight.setPhase(node_id, new_phase)