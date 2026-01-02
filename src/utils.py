import matplotlib.pyplot as plt

def plot_results(history_steps, history_rewards, filename="training_result.png"):
    """
    Plots the cumulative reward over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_steps, history_rewards, label="Cumulative Reward")
    plt.title("RL Agent Performance (Q-Learning)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    print(f"Graph saved to {filename}")
    # plt.show() # Uncomment if running locally with a monitor