#already working

import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom modules
from src.environments.cityflow_env import CityFlowPressLightEnv
from src.algorithms.presslight import PressLightAgent

#we wrote a "Command Line Interface" (CLI). This allows you to test the New York map or the Jinan map without 
#ever touching the code—you just change the text in your terminal.
def parse_args():
    parser = argparse.ArgumentParser(description="Train PressLight Agent on CityFlow")
    
    # Paths
    parser.add_argument("--roadnet", type=str, default="data/NewYork/roadnet_16_1.json",
                        help="Path to road network JSON file")
    parser.add_argument("--flow", type=str, default="data/NewYork/anon_16_1_300_newyork_real_1.json",
                        help="Path to flow JSON file")
    parser.add_argument("--intersection", type=str, default="intersection_1_1",
                        help="ID of the intersection to control")
    
    # Training Hyperparameters
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--steps_per_episode", type=int, default=3600, help="Simulation steps per episode")
    parser.add_argument("--batch_size", type=int, default=32, help="DQN batch size")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    
    return parser.parse_args()

def ensure_directories(base_dir):
    """Creates necessary subdirectories for logs:For the raw numbers, 
    models: To save the .pth files (the trained brains), 
    and figures: For the training graphs"""
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)

def plot_results(rewards, losses, save_dir):
    """Generates and saves training curves."""
    plt.figure(figsize=(12, 5))

    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward (Negative Pressure)")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.legend()

    # Plot Losses
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Avg Loss", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, "figures", f"training_curve_{timestamp}.png"))
    print(f"Plot saved to {os.path.join(save_dir, 'figures')}")
    plt.close()

def main():
    args = parse_args()
    ensure_directories(args.save_dir)
    
    # 1. Initialize Environment
    print(f"Initializing Environment with {args.roadnet}...")
    try:
        env = CityFlowPressLightEnv(
            roadnet_file=args.roadnet,
            flow_file=args.flow,
            intersection_id=args.intersection,
            num_steps=args.steps_per_episode
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your file paths in the 'data' folder.")
        return

    print(f"State Dim: {env.state_dim}, Action Dim: {env.num_phases}")

    # 2. Initialize Agent
    agent = PressLightAgent(
        state_dim=env.state_dim,
        action_dim=env.num_phases,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon=1.0,            # Start with full exploration
        epsilon_decay=1.0,    # Decay per update
        epsilon_min=0.01
    )

    # Logging setup
    log_file_path = os.path.join(args.save_dir, "logs", "training_log.csv")
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Reward', 'Avg Loss', 'Epsilon', 'Steps'])

    episode_rewards = []
    episode_losses = []

    # 3. Training Loop
    print("\nStarting Training...")
    print("-" * 60)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        losses = []
        done = False
        step_count = 0

        while not done:
            # Select Action
            action = agent.get_action(state)
            
            # Step Environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train Agent
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            step_count += 1

        # Calculate metrics
        avg_loss = np.mean(losses) if losses else 0
        
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        # Print Progress
        print(f"Episode {episode}/{args.episodes} | "
              f"Reward: {total_reward:.2f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.4f}")
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= 0.97
        # Log to CSV
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, avg_loss, agent.epsilon, step_count])

        # Save Model periodically (every 10 episodes and the last one)
        if episode % 10 == 0 or episode == args.episodes:
            model_path = os.path.join(args.save_dir, "models", f"presslight_ep{episode}.pth")
            agent.save_model(model_path)

    print("-" * 60)
    print("Training Complete.")
    
    # 4. Generate Final Plots
    plot_results(episode_rewards, episode_losses, args.save_dir)

if __name__ == "__main__":
    main()