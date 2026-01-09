import traci
import time
import sys
import os
import csv
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from datetime import datetime
from src import config, environment, agent, vision

def save_log_entry(writer, step, reward, queue, wait, peds, cars, amb, trucks):
    """Saves data to CSV for the Web UI to read later."""
    writer.writerow({
        "step": step,
        "reward": reward,
        "queue": queue,
        "avg_wait": wait,
        "peds": peds,
        "cars": cars,
        "ambulance": amb,
        "trucks": trucks,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def create_dashboard(step, total_steps, reward, queue, vehicles, vision_data, agent_states):
    """Creates the Rich Table structure."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="vision", size=3)
    )
    
    # Header
    layout["header"].update(Panel(f"🚦 Intelligent Traffic Control | Step: {step}/{total_steps} | Vehicles: {vehicles}", style="bold green"))
    
    # Agent Table
    table = Table(title="🚦 Traffic Light Status")
    table.add_column("Node ID", style="cyan")
    table.add_column("Phase", justify="center")
    table.add_column("Queue", justify="right")
    table.add_column("Occupancy", justify="right")
    table.add_column("Action", justify="center")
    
    for tl_id, data in agent_states.items():
        # Color code actions: Switching phase is yellow, keeping is green
        action_style = "yellow" if data['action'] == 1 else "green"
        action_text = "SWITCH 🔄" if data['action'] == 1 else "HOLD ⏱️"
        table.add_row(
            tl_id, 
            str(data['phase']), 
            str(data['queue']), 
            f"{data['occ']:.1%}", 
            f"[{action_style}]{action_text}[/]"
        )
    layout["main"].update(table)
    
    # Vision Footer
    p, c, a, t = vision_data
    vis_text = f"📷 VISION: Pedestrians: {p} | 🚗 Cars: {c} | 🚑 Ambulances: {a} | 🚛 Trucks: {t}"
    style = "bold red" if a > 0 else "blue"
    layout["vision"].update(Panel(vis_text, style=style))
    
    return layout

def main():
    # Setup CSV Logging for the Web UI
    log_file = open("simulation_data.csv", "w", newline='')
    fieldnames = ["step", "reward", "queue", "avg_wait", "peds", "cars", "ambulance", "trucks", "timestamp"]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    # ... [Keep your existing Init code: Folders, SUMO start, Vision, Agents] ...
    # (Assuming standard setup code from your file is here)
    environment.start_simulation(use_gui=True)
    eye = vision.TrafficEye()
    agents = {tl: agent.TrafficAgent(len(config.DETECTOR_GROUPS[tl]), tl) for tl in config.TRAFFIC_LIGHT_IDS}
    
    # Initialize Tracking
    last_switch = {tl: -config.STANDARD_MIN_GREEN for tl in config.TRAFFIC_LIGHT_IDS}
    phase_nums = {tl: 6 for tl in config.TRAFFIC_LIGHT_IDS} # Simplified for snippet
    
    print("Starting Live Dashboard...")
    
    # --- RICH LIVE CONTEXT ---
    with Live(refresh_per_second=4) as live:
        for step in range(config.TOTAL_STEPS):
            traci.simulationStep()
            
            # ALWAYS use Traci for accurate counts
            queue_data, length_data = environment.get_vehicle_counts_with_lengths()
            occupancy_data = environment.get_vehicle_occupancy()
            avg_wait_time = environment.get_average_waiting_time()
            total_vehicles = environment.get_vehicle_counts()
            
            # Ground Truth for Agents (Fast & Accurate)
            peds, cars, amb, trucks = eye.get_traci_ground_truth()
            
            # 2. Visualization (Showoff Source)
            # ---------------------------------------------
            # Only run image processing every 50 steps to save screenshot
            if step % 50 == 0:
                 # We don't use the return values here, just trigger the save
                 eye.get_vision_reward(step=step, debug=True)

            # 3. Agent Decision
            # ---------------------------------------------
            agent_states = {}
            total_reward = 0
            
            for tl_id in config.TRAFFIC_LIGHT_IDS:
                det_ids = config.DETECTOR_GROUPS[tl_id]
                queues = [queue_data.get(d, 0) for d in det_ids]
                occs = [occupancy_data.get(d, 0) for d in det_ids]
                
                # Check actual light state
                current_phase_idx = traci.trafficlight.getPhase(tl_id)
                current_phase_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                
                state = agents[tl_id].get_state(queues, current_phase_idx)
                
                # --- EMERGENCY OVERRIDE (Using Accurate TRACI Count) ---
                if amb > 0:
                    action = agents[tl_id].choose_emergency_action(current_phase_state)
                    min_green = 0 # Immediate switch
                else:
                    action = agents[tl_id].choose_action(state)
                    min_green = config.STANDARD_MIN_GREEN
                # -------------------------------------------------------

                # Execute
                if action == 1 and (step - last_switch[tl_id] >= min_green):
                    environment.set_traffic_light_phase(tl_id, action)
                    last_switch[tl_id] = step
                
                agent_states[tl_id] = {
                    'phase': current_phase_idx,
                    'queue': sum(queues),
                    'occ': np.mean(occs) if occs else 0,
                    'action': action
                }
            
            # 4. Update Dashboard (Using Accurate TRACI Data)
            # ---------------------------------------------
            dashboard = create_dashboard(
                step, config.TOTAL_STEPS, total_reward, 
                sum(queue_data.values()), total_vehicles, 
                (peds, cars, amb, trucks), agent_states
            )
            live.update(dashboard)
            
            save_log_entry(writer, step, total_reward, sum(queue_data.values()), avg_wait_time, peds, cars, amb, trucks)
            log_file.flush()
            time.sleep(0.05)

    log_file.close()
    environment.close_simulation()

if __name__ == "__main__":
    main()


    # Penality Ambulance
    # Temps d'attente in dash
    # screenshots for opecv (ambulance in white and camions in blue bus yellow car ...)
    # add yolov8