import traci
import time
import sys
import numpy as np
from src import config, environment, agent, vision, utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("🚦 Starting Intelligent Traffic System (Enhanced Version with Vehicle Length)...")
    print("="*60)
    print(f"Using configuration from: {config.DATA_DIR}/networks")
    print(f"Traffic Lights: {config.TRAFFIC_LIGHT_IDS}")
    print(f"Detectors: {sum(len(g) for g in config.DETECTOR_GROUPS.values())} grouped across {len(config.TRAFFIC_LIGHT_IDS)} nodes")
    print(f"Total steps: {config.TOTAL_STEPS}")
    print("="*60)
    
    try:
        environment.start_simulation(use_gui=True)
        time.sleep(5)  # Give more time for GUI to stabilize
    except Exception as e:
        print(f"Error starting simulation: {e}")
        print("Make sure SUMO is properly installed and SUMO_HOME is set")
        sys.exit(1)
    
    # Initialize components
    eye = vision.TrafficEye()
    
    # MODIFICATION: Initialiser les agents avec tl_id
    agents = {tl: agent.TrafficAgent(len(config.DETECTOR_GROUPS[tl]), tl) 
              for tl in config.TRAFFIC_LIGHT_IDS}
    
    # Stockage des données de vision
    last_vision_data = {tl: (0, 0, 0, 0) for tl in config.TRAFFIC_LIGHT_IDS}
    global_vision_data = (0, 0, 0, 0)
    
    # Variables de coordination
    last_global_reward = 0
    consecutive_bad_rewards = 0
    emergency_mode = False
    emergency_steps_remaining = 0
    
    # Get phase counts dynamically
    phase_nums = {}
    for tl in config.TRAFFIC_LIGHT_IDS:
        try:
            defn = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
            phase_nums[tl] = len(defn.phases)
        except:
            phase_nums[tl] = 6  # Default fallback
    
    last_switch = {tl: -config.STANDARD_MIN_GREEN for tl in config.TRAFFIC_LIGHT_IDS}
    
    # Historiques améliorés
    history_rewards = []
    history_steps = []
    history_queues = []
    history_lengths = []  # AJOUT: Historique des longueurs
    history_vehicles = []
    history_waiting_times = []
    history_occupancy = []  # AJOUT: Historique d'occupation
    history_actions = {tl: [] for tl in config.TRAFFIC_LIGHT_IDS}
    history_phases = {tl: [] for tl in config.TRAFFIC_LIGHT_IDS}
    
    cumulative_reward = 0
    start_time = time.time()
    
    # Main Loop
    step = 0
    try:
        for step in range(config.TOTAL_STEPS):
            traci.simulationStep()
            
            # --- A. Perception AMÉLIORÉE avec longueurs ---
            # AVANT: queue_data = environment.get_queue_lengths()
            # APRÈS: Récupérer nombre, longueur ET occupation
            queue_data, length_data = environment.get_vehicle_counts_with_lengths()
            occupancy_data = environment.get_vehicle_occupancy()
            
            total_queue = sum(queue_data.values())
            total_length = sum(length_data.values())  # Longueur totale en mètres
            avg_occupancy = np.mean(list(occupancy_data.values())) if occupancy_data else 0
            
            # --- B. Vision Analysis ---
            VISION_FREQUENCY = config.VISION_FREQUENCY
            DEBUG_VISION_FREQUENCY = 50
            
            do_vision_analysis = (step % VISION_FREQUENCY == 0) or (step == 0)
            do_detailed_vision = (step % DEBUG_VISION_FREQUENCY == 0)
            
            if do_vision_analysis:
                vision_reward, peds, cars, ambulances, trucks = eye.get_vision_reward(
                    step=step, 
                    debug=do_detailed_vision
                )
                global_vision_data = (peds, cars, ambulances, trucks)
                
                # Mettre à jour les données par feu (simplifié - même données pour tous)
                for tl_id in config.TRAFFIC_LIGHT_IDS:
                    last_vision_data[tl_id] = global_vision_data
            else:
                vision_reward = 0
                peds, cars, ambulances, trucks = global_vision_data
            
            num_tl = len(config.TRAFFIC_LIGHT_IDS)
            global_reward = vision_reward / num_tl if num_tl > 0 else 0
            
            # --- C. Emergency Handling ---
            current_min_green = config.STANDARD_MIN_GREEN
            
            if ambulances > 0:
                if not emergency_mode:
                    print(f"\n!!! EMERGENCY DETECTED at Step {step} !!! Priority Override.")
                    emergency_mode = True
                    emergency_steps_remaining = 20  # Maintenir le mode urgence pour 20 steps
                
                current_min_green = 0
                emergency_steps_remaining -= 1
                
                if emergency_steps_remaining <= 0:
                    emergency_mode = False
                    print("Emergency mode ended.")
            
            # --- D. Action Execution & Learning AMÉLIORÉE ---
            total_reward = 0
            actions = {}
            new_phases = {}
            
            # Coordination simple entre feux
            should_coordinate = (config.ENABLE_COORDINATION and 
                                step > 100 and 
                                step % config.COORDINATION_FREQUENCY == 0 and
                                not emergency_mode)
            
            # Calculer la densité globale pour décision de coordination
            if should_coordinate:
                total_vehicles_count = environment.get_vehicle_counts()
                global_density = total_queue / max(1, total_vehicles_count) if total_vehicles_count > 0 else 0
                should_coordinate = (global_density > 0.6)
            
            for tl_id in config.TRAFFIC_LIGHT_IDS:
                det_ids = config.DETECTOR_GROUPS[tl_id]
                queues = [queue_data.get(det_id, 0) for det_id in det_ids]
                lengths = [length_data.get(det_id, 0.0) for det_id in det_ids]  # AJOUT: Longueurs
                occupancies = [occupancy_data.get(det_id, 0.0) for det_id in det_ids]  # AJOUT: Occupation
                
                current_phase = traci.trafficlight.getPhase(tl_id)
                
                # État enrichi avec TOUTES les données
                state = agents[tl_id].get_state(
                    queues, 
                    current_phase % phase_nums[tl_id],
                    vision_data=last_vision_data[tl_id],
                    vehicle_lengths=lengths,  # AJOUT
                    occupancy=occupancies     # AJOUT
                )
                
                # Choix d'action (avec éventuelle surcharge en mode urgence)
                if emergency_mode and ambulances > 0:
                    # En mode urgence, forcer le changement si possible
                    can_switch = (step - last_switch[tl_id] >= current_min_green)
                    action = 1 if can_switch else 0
                else:
                    action = agents[tl_id].choose_action(state)
                
                actions[tl_id] = action
                
                # Vérifier si on peut changer
                can_switch = (step - last_switch[tl_id] >= current_min_green)
                
                if action == 1 and can_switch:
                    # Coordination: alterner les groupes de feux
                    if should_coordinate:
                        # Groupe 1: Node1, Node3
                        # Groupe 2: Node2, Node5, Node7
                        if tl_id in ['Node1', 'Node3'] and step % 2 == 0:
                            new_phase = environment.set_traffic_light_phase(tl_id, action)
                            last_switch[tl_id] = step
                        elif tl_id in ['Node2', 'Node5', 'Node7'] and step % 2 == 1:
                            new_phase = environment.set_traffic_light_phase(tl_id, action)
                            last_switch[tl_id] = step
                        else:
                            new_phase = current_phase
                    else:
                        new_phase = environment.set_traffic_light_phase(tl_id, action)
                        last_switch[tl_id] = step
                    
                    if do_detailed_vision and new_phase != current_phase:
                        print(f"  {tl_id}: Phase {current_phase} → {new_phase}")
                else:
                    new_phase = current_phase
                
                new_phases[tl_id] = new_phase
                
                # Nouvel état
                new_queues = [queue_data.get(det_id, 0) for det_id in det_ids]
                new_lengths = [length_data.get(det_id, 0.0) for det_id in det_ids]  # AJOUT
                new_occupancies = [occupancy_data.get(det_id, 0.0) for det_id in det_ids]  # AJOUT
                new_phase_actual = traci.trafficlight.getPhase(tl_id)
                
                # ============================================
                # MODIFICATION: Récompense AMÉLIORÉE avec longueur et occupation
                # ============================================
                
                # 1. Pénalité pour nombre de véhicules (légère)
                queue_penalty = -sum(queues) * 0.2
                
                # 2. AJOUT: Pénalité pour longueur des véhicules (plus réaliste)
                length_penalty = -sum(lengths) * 0.01  # 0.01 par mètre de véhicule
                
                # 3. AJOUT: Pénalité pour occupation critique
                occupancy_penalty = 0
                critical_occupancy = 0
                for occ in occupancies:
                    if occ > 80.0:  # Occupation critique (>80%)
                        occupancy_penalty -= 2.0
                        critical_occupancy += 1
                    elif occ > 60.0:  # Occupation élevée
                        occupancy_penalty -= 0.5
                
                # 4. AJOUT: Récompense pour avoir vidé le détecteur
                clearing_reward = 0
                if sum(queues) == 0:
                    clearing_reward = 1.0
                
                # 5. Récompense de stabilité
                stability_reward = config.REWARD_STABILITY if action == 0 else -config.REWARD_STABILITY
                
                # 6. AJOUT: Récompense pour réduction de la longueur
                length_reduction_reward = 0
                if step > 0:
                    # Si la longueur totale a diminué depuis le dernier état
                    prev_length = agents[tl_id].last_total_length if hasattr(agents[tl_id], 'last_total_length') else sum(lengths)
                    if sum(lengths) < prev_length:
                        length_reduction_reward = (prev_length - sum(lengths)) * 0.02
                
                # Stocker la longueur actuelle pour la prochaine itération
                agents[tl_id].last_total_length = sum(lengths)
                
                # 7. Pénalité d'urgence si ambulance détectée mais pas priorisée
                emergency_penalty = 0.0
                if ambulances > 0 and action == 0 and can_switch and not emergency_mode:
                    emergency_penalty = config.REWARD_EMERGENCY * 0.2
                
                # Calcul de la récompense totale
                reward = (queue_penalty + length_penalty + occupancy_penalty + 
                         clearing_reward + stability_reward + length_reduction_reward + 
                         emergency_penalty + global_reward)
                
                # Nouvel état avec toutes les données
                new_state = agents[tl_id].get_state(
                    new_queues, 
                    new_phase_actual % phase_nums[tl_id],
                    vision_data=last_vision_data[tl_id],
                    vehicle_lengths=new_lengths,
                    occupancy=new_occupancies
                )
                
                # Apprentissage (sauf en mode urgence forcé)
                if not (emergency_mode and ambulances > 0):
                    agents[tl_id].learn(state, action, reward, new_state)
                
                total_reward += reward
                history_actions[tl_id].append(action)
                history_phases[tl_id].append(new_phase_actual)
            
            # --- E. Logging & Statistics AMÉLIORÉES ---
            cumulative_reward += total_reward
            history_rewards.append(total_reward)
            history_steps.append(step)
            history_queues.append(total_queue)
            history_lengths.append(total_length)  # AJOUT
            history_occupancy.append(avg_occupancy)  # AJOUT
            
            current_vehicles = environment.get_vehicle_counts()
            history_vehicles.append(current_vehicles)
            
            avg_waiting = environment.get_average_waiting_time()
            history_waiting_times.append(avg_waiting)
            
            # Logging détaillé
            LOG_FREQUENCY = config.LOG_FREQUENCY
            if step % LOG_FREQUENCY == 0 or do_detailed_vision:
                if do_detailed_vision:
                    print(f"\n{'='*60}")
                    print(f"Step {step:4d} | Queue: {total_queue:3d} | Length: {total_length:6.1f}m | Occupancy: {avg_occupancy:5.1f}%")
                    print(f"Vehicles: {current_vehicles:3d} | Wait: {avg_waiting:5.1f}s | Vision: {peds:2d}P {cars:2d}C {ambulances:2d}A {trucks:2d}T")
                    print(f"Reward: {total_reward:7.1f} | Cum: {cumulative_reward:7.1f}")
                    
                    if emergency_mode:
                        print(f"EMERGENCY MODE ACTIVE ({emergency_steps_remaining} steps remaining)")
                    
                    # Détails par feu (limité à 3 pour lisibilité)
                    for tl_id in config.TRAFFIC_LIGHT_IDS[:3]:
                        det_ids = config.DETECTOR_GROUPS[tl_id][:2]  # 2 premiers détecteurs
                        local_queue = sum(queue_data.get(det_id, 0) for det_id in det_ids)
                        local_length = sum(length_data.get(det_id, 0.0) for det_id in det_ids)
                        local_occupancy = np.mean([occupancy_data.get(det_id, 0.0) for det_id in det_ids])
                        
                        action_history = history_actions[tl_id][-5:] if history_actions[tl_id] else []
                        action_rate = sum(action_history) / len(action_history) if action_history else 0
                        
                        print(f"  {tl_id}: Phase={new_phases[tl_id]:2d}, Queue={local_queue:3d}, "
                              f"Length={local_length:5.1f}m, Occupancy={local_occupancy:5.1f}%, "
                              f"Action={actions[tl_id]}, Rate={action_rate:.2f}")
                    
                    print(f"{'='*60}")
                else:
                    # Log minimal
                    if step % 50 == 0:  # Tous les 50 pas
                        print(f"Step {step:4d} | Queue: {total_queue:3d} | Length: {total_length:6.1f}m | "
                              f"Vehicles: {current_vehicles:3d} | Reward: {total_reward:6.1f}")
            
            # --- F. Performance Control ---
            # Contrôle de vitesse adaptatif
            if current_vehicles > 50:
                time.sleep(0.1)
            elif current_vehicles > 30:
                time.sleep(0.075)
            else:
                time.sleep(0.05)
            
            # --- G. Early Stopping Detection ---
            if step > 100 and len(history_rewards) >= 20:
                recent_rewards = history_rewards[-20:]
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                
                if abs(avg_recent) < 1.0 and abs(total_reward) < 2.0:
                    consecutive_bad_rewards += 1
                else:
                    consecutive_bad_rewards = 0
                
                # Arrêt prématuré si stagnation prolongée
                if consecutive_bad_rewards > 50:
                    print(f"\n⚠️ Early stop: Stagnation detected at step {step}")
                    break
            
            # Petite pause pour stabilité
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                steps_per_second = step / elapsed
                print(f"Progress: {step}/{config.TOTAL_STEPS} steps ({step/config.TOTAL_STEPS*100:.1f}%) | Speed: {steps_per_second:.1f} steps/sec")
            
    except traci.exceptions.FatalTraCIError as e:
        print(f"\nSUMO Simulation error: {e}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error at step {step}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- H. Final Cleanup & Results AMÉLIORÉES ---
        print("\n" + "="*60)
        print("=== Simulation Ended (with Vehicle Length Metrics) ===")
        print("="*60)
        
        total_elapsed = time.time() - start_time
        
        print(f"\n📊 Performance Summary:")
        print(f"  Total steps completed: {step}")
        print(f"  Total time: {total_elapsed:.1f} seconds")
        print(f"  Steps per second: {step/max(1, total_elapsed):.1f}")
        print(f"  Final cumulative reward: {cumulative_reward:.1f}")
        print(f"  Average reward per step: {cumulative_reward/max(1, step):.3f}")
        
        if history_queues and history_lengths:
            avg_queue = sum(history_queues) / len(history_queues)
            max_queue = max(history_queues)
            avg_length = sum(history_lengths) / len(history_lengths)
            max_length = max(history_lengths)
            
            print(f"\n🚗 Traffic Statistics:")
            print(f"  Average queue length: {avg_queue:.1f} vehicles")
            print(f"  Maximum queue length: {max_queue} vehicles")
            print(f"  Average vehicle length: {avg_length:.1f} meters")
            print(f"  Maximum vehicle length: {max_length:.1f} meters")
            
            # Ratio longueur/nombre (indicateur de type de trafic)
            if avg_queue > 0:
                avg_length_per_vehicle = avg_length / avg_queue
                print(f"  Average length per vehicle: {avg_length_per_vehicle:.1f}m")
                if avg_length_per_vehicle > 7.0:
                    print(f"  → High proportion of trucks/buses (>7m per vehicle)")
        
        if history_vehicles:
            avg_vehicles = sum(history_vehicles) / len(history_vehicles)
            max_vehicles = max(history_vehicles)
            print(f"  Average vehicles in simulation: {avg_vehicles:.1f}")
            print(f"  Maximum vehicles: {max_vehicles}")
        
        if history_waiting_times:
            avg_waiting = sum(history_waiting_times) / len(history_waiting_times)
            max_waiting = max(history_waiting_times)
            print(f"  Average waiting time: {avg_waiting:.1f}s")
            print(f"  Maximum waiting time: {max_waiting:.1f}s")
        
        if history_occupancy:
            avg_occupancy = sum(history_occupancy) / len(history_occupancy)
            max_occupancy = max(history_occupancy)
            print(f"  Average detector occupancy: {avg_occupancy:.1f}%")
            print(f"  Maximum detector occupancy: {max_occupancy:.1f}%")
        
        # Statistiques par agent
        print(f"\n🤖 Agent Statistics:")
        for tl_id, ag in agents.items():
            stats = ag.get_stats() if hasattr(ag, 'get_stats') else {}
            q_size = stats.get('q_table_size', 0)
            action_0_rate = stats.get('action_0_rate', 0)
            action_1_rate = stats.get('action_1_rate', 0)
            
            # AJOUT: Indicateur de performance basé sur les actions
            performance_indicator = "🟢" if action_1_rate > 0.2 else ("🟡" if action_1_rate > 0.1 else "🔴")
            
            print(f"  {tl_id}: {q_size:3d} states | "
                  f"Actions: {action_0_rate:.1%} keep, {action_1_rate:.1%} change {performance_indicator}")
        
        # Vision statistics
        vision_stats = eye.get_stats() if hasattr(eye, 'get_stats') else {}
        print(f"\n👁️ Vision Statistics:")
        print(f"  YOLO active: {vision_stats.get('use_yolo', False)}")
        print(f"  Total detections: {vision_stats.get('total_detections', 0)}")
        if 'detection_counts' in vision_stats:
            counts = vision_stats['detection_counts']
            print(f"  Breakdown: {counts.get('peds', 0)} peds, {counts.get('cars', 0)} cars, "
                  f"{counts.get('ambulances', 0)} ambulances, {counts.get('trucks', 0)} trucks")
        
        try:
            environment.close_simulation()
        except:
            pass
        
        # Save Q-tables
        print(f"\n💾 Saving Q-tables...")
        import pickle
        for tl_id, ag in agents.items():
            filename = f"q_table_{tl_id}.pkl"
            ag.save_q_table(filename)
        
        # Plot results
        print(f"\n📈 Generating plots...")
        if history_steps and history_rewards:
            utils.plot_results(history_steps, history_rewards, filename="training_result.png")
            
            # Plot avancé
            try:
                utils.plot_advanced_results(agents, filename="advanced_results.png")
            except Exception as e:
                print(f"Could not create advanced plots: {e}")
            
            # Sauvegarde des métriques
            try:
                utils.save_training_metrics(history_steps, history_rewards, agents, filename="training_metrics.json")
            except Exception as e:
                print(f"Could not save metrics: {e}")
            
            # Plot supplémentaire avec les nouvelles métriques
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 12))
                
                plt.subplot(3, 3, 1)
                plt.plot(history_steps, history_rewards)
                plt.title("Reward over time")
                plt.xlabel("Step")
                plt.ylabel("Reward")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 2)
                plt.plot(history_steps, history_queues)
                plt.title("Queue length (vehicles)")
                plt.xlabel("Step")
                plt.ylabel("Vehicles")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 3)
                plt.plot(history_steps, history_lengths)
                plt.title("Total vehicle length")
                plt.xlabel("Step")
                plt.ylabel("Length (m)")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 4)
                plt.plot(history_steps, history_occupancy)
                plt.title("Detector occupancy")
                plt.xlabel("Step")
                plt.ylabel("Occupancy (%)")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 5)
                plt.plot(history_steps, history_vehicles)
                plt.title("Total vehicles in simulation")
                plt.xlabel("Step")
                plt.ylabel("Vehicles")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 6)
                plt.plot(history_steps, history_waiting_times)
                plt.title("Average waiting time")
                plt.xlabel("Step")
                plt.ylabel("Waiting time (s)")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 7)
                # Actions pour les 3 premiers feux
                colors = ['red', 'green', 'blue']
                for i, tl_id in enumerate(config.TRAFFIC_LIGHT_IDS[:3]):
                    if history_actions[tl_id]:
                        plt.plot(history_steps[-len(history_actions[tl_id]):], 
                                history_actions[tl_id], 
                                label=tl_id, 
                                color=colors[i % len(colors)],
                                alpha=0.7)
                plt.title("Actions over time")
                plt.xlabel("Step")
                plt.ylabel("Action (0=keep, 1=change)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 8)
                # Phases pour les 3 premiers feux
                for i, tl_id in enumerate(config.TRAFFIC_LIGHT_IDS[:3]):
                    if history_phases[tl_id]:
                        plt.plot(history_steps[-len(history_phases[tl_id]):], 
                                history_phases[tl_id], 
                                label=tl_id, 
                                color=colors[i % len(colors)],
                                alpha=0.7)
                plt.title("Phases over time")
                plt.xlabel("Step")
                plt.ylabel("Phase")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 9)
                # Ratio longueur/nombre
                if len(history_queues) == len(history_lengths):
                    ratios = []
                    for q, l in zip(history_queues, history_lengths):
                        if q > 0:
                            ratios.append(l / q)
                        else:
                            ratios.append(0)
                    
                    plt.plot(history_steps, ratios, color='purple')
                    plt.title("Average length per vehicle")
                    plt.xlabel("Step")
                    plt.ylabel("Length/vehicle (m)")
                    plt.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='Car (5m)')
                    plt.axhline(y=12.0, color='red', linestyle='--', alpha=0.5, label='Truck (12m)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig("detailed_analysis_with_length.png", dpi=150)
                print("Detailed analysis plot saved to detailed_analysis_with_length.png")
                plt.close()
            except Exception as e:
                print(f"Could not create detailed plots: {e}")
        
        print(f"\n📁 Debug screenshots saved to: {eye.debug_folder}/")
        print("(Keeping only the most recent 25 debug files)")
        
        print("\n" + "="*60)
        print("✅ Simulation Completed Successfully with Vehicle Length Tracking!")
        print("="*60)

if __name__ == "__main__":
    main()