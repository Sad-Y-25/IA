import traci
import time
import csv
import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from datetime import datetime
from src import config, environment, vision

# --- 1. CONFIGURATION CLASSIQUE ---
# Temps fixe pour chaque phase (ex: le feu change toutes les 40 étapes)
PHASE_DURATION = 40 

def save_log_entry(writer, step, reward, queue, wait, peds, cars, amb, trucks):
    """Sauvegarde identique pour que le dashboard puisse lire le fichier."""
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

def create_dashboard(step, total_steps, queue, vehicles):
    """Dashboard simplifié pour le mode Classique."""
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="main", ratio=1))
    layout["header"].update(Panel(f"⏱️ MODE CLASSIQUE (Timer) | Step: {step}/{total_steps} | Véhicules: {vehicles}", style="bold red"))
    return layout

def main():
    # --- 2. NOM DU FICHIER DE SORTIE ---
    # On sauvegarde sous un nom différent pour la comparaison
    log_file = open("resultats_classic.csv", "w", newline='')
    fieldnames = ["step", "reward", "queue", "avg_wait", "peds", "cars", "ambulance", "trucks", "timestamp"]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    # Démarrage SUMO
    environment.start_simulation(use_gui=True)
    
    # On garde la vision juste pour les stats (pas pour décider)
    eye = vision.TrafficEye()
    
    print("Démarrage de la simulation CLASSIQUE (Timer)...")
    
    with Live(refresh_per_second=4) as live:
        for step in range(config.TOTAL_STEPS):
            traci.simulationStep()
            
            # Récupération des données
            queue_data, _ = environment.get_vehicle_counts_with_lengths()
            total_vehicles = environment.get_vehicle_counts()
            
            # Vision (pour les stats seulement)
            if step % 50 == 0:
                _, peds, cars, amb, trucks = eye.get_vision_reward(step=step, debug=False)
                last_vis = (peds, cars, amb, trucks)
            else:
                peds, cars, amb, trucks = last_vis if 'last_vis' in locals() else (0,0,0,0)

            # --- 3. LOGIQUE CLASSIQUE (SANS IA) ---
            # Au lieu de demander à l'agent, on change le feu si le timer est atteint
            action = 0
            if step % PHASE_DURATION == 0:
                action = 1 # Changer de phase (Switch)
            else:
                action = 0 # Garder la phase (Hold)

            # Appliquer l'action aux feux
            for tl_id in config.TRAFFIC_LIGHT_IDS:
                # On applique la même logique simple à tous les feux
                environment.set_traffic_light_phase(tl_id, action)
            
            # --- 4. LOGGING ---
            # On met reward=0 car il n'y a pas d'IA ici, mais on garde la colonne pour le CSV
            total_queue = sum(queue_data.values())
            
            dashboard = create_dashboard(step, config.TOTAL_STEPS, total_queue, total_vehicles)
            live.update(dashboard)
            
            save_log_entry(writer, step, 0, total_queue, 0, peds, cars, amb, trucks)
            log_file.flush()
            
            time.sleep(0.02) # Simulation un peu plus rapide

    log_file.close()
    environment.close_simulation()

if __name__ == "__main__":
    main()