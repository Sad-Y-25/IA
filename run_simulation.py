import traci
from src import config, environment, agent, vision, utils

def main():
    print("🚦 Starting Intelligent Traffic System (Refactored)...")
    
    # 1. Initialize
    environment.start_simulation(use_gui=True)
    brain = agent.TrafficAgent()
    
    history_rewards = []
    history_steps = []
    cumulative_reward = 0
    
    last_switch_step = -config.STANDARD_MIN_GREEN
    
    # 2. Main Loop
    for step in range(config.TOTAL_STEPS):
        traci.simulationStep()
        
        # --- A. Perception ---
        # Get physical sensor data
        queues = environment.get_queue_lengths()
        current_phase = traci.trafficlight.getPhase("Node2") # Ensure Node ID matches your net.xml
        
        # Get State from Agent
        state = brain.get_state(queues, current_phase)
        
        # Decide Action
        action = brain.choose_action(state)
        
        # --- B. Emergency Override (Vision) ---
        # Analyze image for emergency vehicles
        do_debug = (step % 50 == 0)
        _, _, _, n_emerg = vision.get_vision_reward(debug=do_debug)
        
        current_min_green = config.STANDARD_MIN_GREEN
        if n_emerg > 0:
            print(f"!!! EMERGENCY DETECTED at Step {step} !!! Priority Override.")
            current_min_green = 0 # Allow immediate switch
            
        # --- C. Action Execution ---
        # Only switch if enough time has passed (unless emergency)
        if action == 1 and (step - last_switch_step > current_min_green):
            environment.set_traffic_light_phase("Node2", 1) # Action 1 triggers phase change logic
            last_switch_step = step
            
        # --- D. Learning ---
        # Calculate Reward based on Vision (Congestion Penalty)
        reward, n_c, n_h, n_e = vision.get_vision_reward(debug=False)
        
        # Observe new state
        new_queues = environment.get_queue_lengths()
        new_phase = traci.trafficlight.getPhase("Node2")
        new_state = brain.get_state(new_queues, new_phase)
        
        # Update Q-Table
        brain.learn(state, action, reward, new_state)
        
        # Logging
        cumulative_reward += reward
        if step % 10 == 0:
            print(f"Step {step} | Cars: {n_c} | Reward: {reward:.2f}")
            history_steps.append(step)
            history_rewards.append(cumulative_reward)

    # 3. Cleanup & Results
    environment.close_simulation()
    utils.plot_results(history_steps, history_rewards)
    print("✅ Simulation Completed.")

if __name__ == "__main__":
    main()