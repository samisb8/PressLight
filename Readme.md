🚦 PressLight: Adaptive Traffic Signal Control with DQN
📋 Overview
Implementation of a Deep Q-Network (DQN) agent for adaptive traffic signal control that minimizes traffic pressure in urban networks. Achieves 8.10% reduction in traffic pressure compared to fixed-time controllers.

🚀 Quick Start
Option 1: Docker (Recommended - One Command)
bash
# Builds the Docker image and runs everything automatically
docker-compose up
What happens when you run this:

Builds the Docker image from Dockerfile

Starts a container named rl_presslight_training_2

Mounts volumes to save results locally

Runs the training script for 150 episodes

Evaluates the trained model

Saves everything to ./results/ on your computer

Option 2: Step-by-Step Docker
bash
# 1. Build the Docker image manually
docker-compose build

# 2. Run training only
docker-compose run rl-training python -m src.run_experiment --episodes 150

# 3. Run evaluation on saved model
docker-compose run rl-training python -m src.evaluate_comparison --model_path results/models/presslight_ep150.pth
Option 3: Manual Installation (No Docker)
bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install CityFlow simulator
pip install git+https://github.com/cityflow-project/CityFlow.git

# 3. Quick test (50 episodes)
python -m src.run_experiment --episodes 50
📁 Project Structure
text
RL_PROJECT_2/
├── src/
│   ├── algorithms/          # PressLight DQN agent
│   ├── environments/        # CityFlow wrapper
│   └── run_experiment.py   # Main training script
├── data/                   # Traffic networks (NYC, Jinan)
├── results/                # Outputs: logs, models, plots
├── docker-compose.yml      # One-command reproduction
├── Dockerfile             # Container definition
└── requirements.txt       # Python dependencies
🔧 Training Configuration
Hyperparameters (Optimized)
Parameter	Value	Description
Learning Rate	0.00005	Critical for stability
Batch Size	32	Mini-batch for training
Replay Buffer	10,000	Experience memory
Epsilon Decay	0.97	Exploration→exploitation
Network Architecture
text
Input(56) → Dense(128) → ReLU → Dense(64) → ReLU → Output(8)
State dimensions:

Phase encoding: 8 (one-hot)

Incoming lanes: 36 (12 lanes × 3 segments)

Outgoing lanes: 12

📊 Results
9.41% reduction in traffic pressure vs fixed-time controller

All outputs saved to results/ folder:

results/logs/training_log.csv - Training metrics

results/models/presslight_ep*.pth - Model checkpoints

results/figures/training_curve_*.png - Performance plots

🧪 Experiments
Test on different networks:

bash
# NYC network (default)
python -m src.run_experiment --episodes 150

# With custom parameters
python -m src.run_experiment \
    --roadnet data/NewYork/roadnet_16_1.json \
    --flow data/NewYork/anon_16_1_300_newyork_real_1.json \
    --intersection intersection_1_1 \
    --episodes 150 \
    --lr 0.00005
🐳 Docker Details
What's in the Dockerfile?
dockerfile
FROM python:3.9-slim                          # Base image
RUN apt-get update && apt-get install ...     # System dependencies
COPY requirements.txt .                       # Python dependencies
RUN pip install -r requirements.txt           # Install Python packages
RUN pip install git+https://.../CityFlow.git  # Install CityFlow
COPY . .                                      # Copy project files
RUN mkdir -p results/logs...                  # Create output directories
CMD ["sh", "-c", "python -m ..."]            # Default command
Docker Commands Cheatsheet
bash
# Build the image
docker-compose build

# Run the full pipeline (train + evaluate)
docker-compose up

# Run only training
docker run -v $(pwd)/results:/app/results -v $(pwd)/data:/app/data presslight-rl python -m src.run_experiment --episodes 150

# Enter container shell
docker-compose run rl-training bash

# Clean up
docker-compose down
docker system prune -a
🔍 Key Findings
Learning rate critical: Values above 0.0001 caused instability

State normalization required: Prevents gradient explosion

Segmented lanes: Provides spatial awareness for better decisions

Pressure-based rewards: Enables interpretable optimization

📝 Citation
Based on: Wei et al. "PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network" (KDD 2019)

👤 Authors
Ayoub EL ASSIOUI 

Sami BACHTAOUI - bachtaousami@gmail.com