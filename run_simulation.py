
import traci
import time
import sys
from src import config, environment, agent, vision, utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("🚦 Starting Intelligent Traffic System (Complex Config)...")
    print(f"Using configuration from: {config.DATA_DIR}/networks")
    print(f"Traffic Lights: {config.TRAFFIC_LIGHT_IDS}")
    print(f"Detectors: {sum(len(g) for g in config.DETECTOR_GROUPS.values())} grouped across {len(config.TRAFFIC_LIGHT_IDS)} nodes")
    
    try:
        environment.start_simulation(use_gui=True)
        time.sleep(5)  # Give more time for GUI to stabilize
    except Exception as e:
        print(f"Error starting simulation: {e}")
        print("Make sure SUMO is properly installed and SUMO_HOME is set")
        sys.exit(1)
    
    # Initialize components
    eye = vision.TrafficEye()
    agents = {tl: agent.TrafficAgent(len(config.DETECTOR_GROUPS[tl])) for tl in config.TRAFFIC_LIGHT_IDS}
    
    # Get phase counts dynamically
    phase_nums = {}
    for tl in config.TRAFFIC_LIGHT_IDS:
        try:
            defn = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
            phase_nums[tl] = len(defn.phases)
        except:
            phase_nums[tl] = 6  # Default fallback
    
    last_switch = {tl: -config.STANDARD_MIN_GREEN for tl in config.TRAFFIC_LIGHT_IDS}
    
    history_rewards = []
    history_steps = []
    cumulative_reward = 0
    
    # Main Loop
    step = 0
    try:
        for step in range(config.TOTAL_STEPS):
            traci.simulationStep()
            
            # --- A. Perception ---
            queue_data = environment.get_queue_lengths()
            total_queue = sum(queue_data.values())
            
            # --- B. Vision Analysis ---
            do_detailed_vision = (step % 50 == 0)
            vision_reward, peds, cars, ambulances, trucks = eye.get_vision_reward(step=step, debug=do_detailed_vision)
            
            num_tl = len(config.TRAFFIC_LIGHT_IDS)
            global_reward = vision_reward / num_tl if num_tl > 0 else 0
            
            # Emergency override
            current_min_green = config.STANDARD_MIN_GREEN
            if ambulances > 0:
                print(f"!!! EMERGENCY DETECTED at Step {step} !!! Priority Override.")
                current_min_green = 0
            
            # --- C. Action Execution & Learning ---
            total_reward = 0
            actions = {}  # Track actions per TL
            for tl_id in config.TRAFFIC_LIGHT_IDS:
                det_ids = config.DETECTOR_GROUPS[tl_id]
                queues = [queue_data.get(det_id, 0) for det_id in det_ids]
                
                current_phase = traci.trafficlight.getPhase(tl_id)
                state = agents[tl_id].get_state(queues, current_phase % phase_nums[tl_id])
                
                action = agents[tl_id].choose_action(state)
                actions[tl_id] = action  # Store action
                
                if action == 1 and (step - last_switch[tl_id] >= current_min_green):
                    new_phase = environment.set_traffic_light_phase(tl_id, action)
                    last_switch[tl_id] = step
                    if do_detailed_vision:
                        print(f"Changed {tl_id} phase to {new_phase}")
                
                # New state (approx, as queues from current step)
                new_queues = [queue_data.get(det_id, 0) for det_id in det_ids]
                new_phase = traci.trafficlight.getPhase(tl_id)
                new_state = agents[tl_id].get_state(new_queues, new_phase % phase_nums[tl_id])
                
                queue_reward = -sum(queues) * 0.5
                reward = queue_reward + global_reward
                
                agents[tl_id].learn(state, action, reward, new_state)
                total_reward += reward
            
            # Logging
            cumulative_reward += total_reward
            if step % 10 == 0:
                total_vehicles = environment.get_vehicle_counts()
                if do_detailed_vision:
                    print(f"Step {step:3d} | Total Queue: {total_queue:3d} | Vision: {peds}p,{cars}c,{ambulances}a,{trucks}t | Vehicles: {total_vehicles:3d} | Reward: {total_reward:6.1f} | Cum: {cumulative_reward:6.1f}")
                    for tl_id in config.TRAFFIC_LIGHT_IDS:
                        current_phase = traci.trafficlight.getPhase(tl_id)
                        local_queue = sum(queue_data.get(det_id, 0) for det_id in config.DETECTOR_GROUPS[tl_id])
                        print(f"  {tl_id}: Phase={current_phase}, Local Queue={local_queue}, Action={actions[tl_id]}")
                else:
                    print(f"Step {step:3d} | Total Queue: {total_queue:3d} | Vehicles: {total_vehicles:3d} | Reward: {total_reward:6.1f} | Cum: {cumulative_reward:6.1f}")
            
            # Small delay
            time.sleep(0.05)
            
    except traci.exceptions.FatalTraCIError as e:
        print(f"\nSUMO Simulation error: {e}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error at step {step}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup & Results
        print("\n=== Simulation Ended ===")
        print(f"Total steps completed: {step}")
        print(f"Final cumulative reward: {cumulative_reward:.1f}")
        
        try:
            environment.close_simulation()
        except:
            pass
        
        # Save Q-tables (one per TL)
        for tl_id, ag in agents.items():
            ag.save_q_table(f"q_table_{tl_id}.npy")
        
        # Plot results
        if history_steps:
            utils.plot_results(history_steps, history_rewards, filename="training_result.png")
        
        print(f"Debug screenshots saved to: {eye.debug_folder}/")
        print("(Keeping only the most recent 25 debug files)")
        print("✅ Simulation Completed.")

if __name__ == "__main__":
    main()