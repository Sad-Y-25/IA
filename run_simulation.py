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
            
            # 1. Perception
            queue_data, length_data = environment.get_vehicle_counts_with_lengths()
            occupancy_data = environment.get_vehicle_occupancy()
            total_vehicles = environment.get_vehicle_counts()
            
            # 2. Vision (Run less often for speed, but log always)
            if step % 50 == 0:
                _, peds, cars, amb, trucks = eye.get_vision_reward(step=step, debug=False)
                last_vis = (peds, cars, amb, trucks)
            else:
                peds, cars, amb, trucks = last_vis if 'last_vis' in locals() else (0,0,0,0)

            # 3. Agents Decision
            agent_states = {}
            total_reward = 0
            
            for tl_id in config.TRAFFIC_LIGHT_IDS:
                det_ids = config.DETECTOR_GROUPS[tl_id]
                queues = [queue_data.get(d, 0) for d in det_ids]
                occs = [occupancy_data.get(d, 0) for d in det_ids]
                
                # ... [Keep your Agent Logic: State, Action, Reward, Learn] ...
                # Simplified for display example:
                current_phase = traci.trafficlight.getPhase(tl_id)
                state = agents[tl_id].get_state(queues, current_phase)
                action = agents[tl_id].choose_action(state)
                
                # Execute Logic
                if action == 1 and (step - last_switch[tl_id] >= config.STANDARD_MIN_GREEN):
                    environment.set_traffic_light_phase(tl_id, action)
                    last_switch[tl_id] = step
                
                # Store for Dashboard
                agent_states[tl_id] = {
                    'phase': current_phase,
                    'queue': sum(queues),
                    'occ': np.mean(occs) if occs else 0,
                    'action': action
                }
            
            # 4. Update Dashboard
            dashboard = create_dashboard(
                step, config.TOTAL_STEPS, total_reward, 
                sum(queue_data.values()), total_vehicles, 
                (peds, cars, amb, trucks), agent_states
            )
            live.update(dashboard)
            
            # 5. Log for Web UI
            save_log_entry(writer, step, total_reward, sum(queue_data.values()), 0, peds, cars, amb, trucks)
            log_file.flush() # Ensure data is written immediately
            
            time.sleep(0.05) # Control simulation speed

    log_file.close()
    environment.close_simulation()

if __name__ == "__main__":
    main()