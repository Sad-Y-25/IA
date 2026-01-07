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
    """Starts the SUMO simulation with the configuration."""
    check_sumo_installation()
    
    sumo_binary = config.SUMO_GUI_BINARY if use_gui else "sumo"
    
    # Get paths for configuration
    sumo_cfg = config.SUMO_CFG_PATH
    network_file = os.path.join(config.DATA_DIR, 'networks', 'RL.net.xml')
    route_file = os.path.join(config.DATA_DIR, 'networks', 'RL.rou.xml')
    additional_file = os.path.join(config.DATA_DIR, 'networks', 'RL.add.xml')
    
    cmd = [
        sumo_binary,
        '-c', sumo_cfg,
        '--net-file', network_file,
        '--route-files', route_file,
        '--additional-files', additional_file,
        '--step-length', config.STEP_LENGTH,
        '--lateral-resolution', '0',
        '--start'  # Auto start
    ]
    
    # Get a free port to avoid conflicts
    port = sumolib.miscutils.getFreeSocketPort()
    traci.start(cmd, port=port)
    
    # Set view to "real world" schema
    traci.gui.setSchema("View #0", "real world")
    
    print(f"Started simulation with configuration")
    print(f"Network: {network_file}")
    print(f"Routes: {route_file}")
    print(f"Additional: {additional_file}")

def close_simulation():
    traci.close()

def get_queue_lengths():
    """Returns a dict of vehicle counts from all induction loops."""
    queue_data = {}
    for group in config.DETECTOR_GROUPS.values():
        for det_id in group:
            try:
                queue_data[det_id] = traci.lanearea.getLastStepVehicleNumber(det_id)
            except:
                queue_data[det_id] = 0  # Detector not found
    return queue_data

def get_all_traffic_light_states():
    """Get current states of all traffic lights."""
    states = {}
    for tl_id in config.TRAFFIC_LIGHT_IDS:
        try:
            states[tl_id] = {
                'phase': traci.trafficlight.getPhase(tl_id),
                'state': traci.trafficlight.getRedYellowGreenState(tl_id),
                'program': traci.trafficlight.getProgram(tl_id)
            }
        except:
            states[tl_id] = None
    return states

def set_traffic_light_phase(node_id, action):
    """
    Simulates switching phases for traffic lights.
    """
    try:
        current_phase = traci.trafficlight.getPhase(node_id)
        defn = traci.trafficlight.getCompleteRedYellowGreenDefinition(node_id)[0]
        num_phases = len(defn.phases)
        if action == 1:  # Switch to next phase
            new_phase = (current_phase + 1) % num_phases
        else:  # Keep current phase
            new_phase = current_phase
        traci.trafficlight.setPhase(node_id, new_phase)
        return new_phase
    except Exception as e:
        print(f"Error setting traffic light phase for {node_id}: {e}")
        return current_phase

def get_vehicle_counts():
    """Get total number of vehicles in simulation."""
    try:
        return traci.vehicle.getIDCount()
    except:
        return 0

def get_average_waiting_time():
    """Calculate average waiting time for all vehicles."""
    try:
        vehicle_ids = traci.vehicle.getIDList()
        if not vehicle_ids:
            return 0
            
        total_waiting = 0
        for veh_id in vehicle_ids:
            total_waiting += traci.vehicle.getWaitingTime(veh_id)
        
        return total_waiting / len(vehicle_ids)
    except:
        return 0
    
def get_vehicle_counts_with_lengths():
    """Returns dicts of vehicle counts and total lengths from all detectors."""
    queue_data = {}
    length_data = {}
    for group in config.DETECTOR_GROUPS.values():
        for det_id in group:
            try:
                queue_data[det_id] = traci.lanearea.getLastStepVehicleNumber(det_id)
                length_data[det_id] = traci.lanearea.getLength(det_id) * queue_data[det_id]  # Approx total length as detector length * count
            except:
                queue_data[det_id] = 0
                length_data[det_id] = 0
    return queue_data, length_data

def get_vehicle_occupancy():
    """Returns dict of occupancy from all detectors (0-1 fraction)."""
    occupancy_data = {}
    for group in config.DETECTOR_GROUPS.values():
        for det_id in group:
            try:
                occupancy_data[det_id] = traci.lanearea.getLastStepOccupancy(det_id)
            except:
                occupancy_data[det_id] = 0.0
    return occupancy_data