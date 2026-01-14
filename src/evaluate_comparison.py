import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import existing modules
from src.environments.cityflow_env import CityFlowPressLightEnv
from src.algorithms.presslight import PressLightAgent

class FixedTimeAgent:
    """
    Simulates a standard traffic light that switches phases 
    every 'duration' seconds cyclically.
    """
    def __init__(self, num_phases, step_len=10, phase_duration=90):
        self.num_phases = num_phases
        self.step_len = step_len     
        self.phase_duration = phase_duration 
        self.current_phase = 0
        self.time_in_phase = 0

    def get_action(self, state=None):
        # Accumulate time
        self.time_in_phase += self.step_len
        
        # If time exceeded duration, switch to next phase
        if self.time_in_phase >= self.phase_duration:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.time_in_phase = 0
            
        return self.current_phase

def run_simulation(env, agent, name):
    """Runs a full episode and returns the pressure history."""
    print(f"Running evaluation for: {name}...")
    state = env.reset()
    done = False
    pressure_history = []
    total_reward = 0

    while not done:
        # Get action
        if name == "FixedTime":
            action = agent.get_action()
        else:
            # RL Agent (No exploration during eval)
            action = agent.get_action(state, training=False)

        # Step
        next_state, reward, done, _ = env.step(action)
        
        # Convert Reward back to Pressure for visualization
        # Reward was: -1 * Pressure / 100
        # So: Pressure = -1 * Reward * 100
        current_pressure = -1 * reward * 100
        pressure_history.append(current_pressure)
        
        total_reward += reward
        state = next_state

    avg_pressure = np.mean(pressure_history)
    print(f"-> {name} Finished. Avg Pressure: {avg_pressure:.2f} | Total Reward: {total_reward:.2f}")
    return pressure_history, avg_pressure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--roadnet", type=str, default="data/NewYork/roadnet_16_1.json")
    parser.add_argument("--flow", type=str, default="data/NewYork/anon_16_1_300_newyork_real_1.json")
    args = parser.parse_args()

    # 1. Initialize Environment
    env = CityFlowPressLightEnv(args.roadnet, args.flow, "intersection_1_1", num_steps=3600)

    # 2. Setup FixedTime Agent (Baseline)
    # 40 seconds per phase is a standard real-world setting
    ft_agent = FixedTimeAgent(env.num_phases, step_len=10, phase_duration=90)

    # 3. Setup PressLight Agent (Your Trained Model)
    rl_agent = PressLightAgent(env.state_dim, env.num_phases)
    
    # Load the weights
    if os.path.exists(args.model_path):
        rl_agent.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model file {args.model_path} not found!")
        return

    # 4. Run Both
    ft_pressures, ft_avg = run_simulation(env, ft_agent, "FixedTime")
    rl_pressures, rl_avg = run_simulation(env, rl_agent, "PressLight (RL)")

    # 5. Plot Comparison
    plt.figure(figsize=(10, 6))
    
    # Plot raw pressure data
    plt.plot(ft_pressures, label=f'FixedTime (Avg: {ft_avg:.1f})', color='grey', alpha=0.7, linestyle='--')
    plt.plot(rl_pressures, label=f'PressLight (Avg: {rl_avg:.1f})', color='blue', linewidth=2)
    
    plt.title("Comparison: Real-World Fixed Timing vs. Trained RL Agent")
    plt.xlabel("Simulation Steps (x10 seconds)")
    plt.ylabel("Intersection Pressure (Queue Imbalance)")
    plt.legend()
    plt.grid(True)
    
    save_path = "results/figures/comparison_result.png"
    plt.savefig(save_path)
    print(f"\nComparison Plot saved to: {save_path}")
    print(f"Improvement: {((ft_avg - rl_avg) / ft_avg) * 100:.2f}% reduction in pressure.")

if __name__ == "__main__":
    main()