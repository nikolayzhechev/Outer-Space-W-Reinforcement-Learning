import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards(rewards, epsilon_values=None):
    def smooth(y, window_size=50):
        return pd.Series(y).rolling(window=window_size, min_periods=1).mean()

    episodes = np.arange(len(rewards))
    
    plt.figure(figsize=(12, 6))

    # Plot raw rewards as scatter
    plt.scatter(episodes, rewards, color='blue', alpha=0.2, label="Raw Reward")

    # Plot smoothed rewards
    plt.plot(episodes, smooth(rewards), label="Smoothed Reward (Rolling Avg)", color='red', linewidth=2)

    # If epsilon values are provided, plot them on a secondary y-axis
    if epsilon_values is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(episodes, epsilon_values, label="Epsilon Decay", color='green', linestyle="dashed")
        ax2.set_ylabel("Epsilon (Exploration Rate)", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()