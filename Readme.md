# 🚦 PressLight: Deep Reinforcement Learning for Adaptive Traffic Signal Control

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Deep Q-Network (DQN) implementation for adaptive traffic signal control that achieves **8.10% reduction in traffic pressure** compared to traditional fixed-time controllers on real NYC traffic data.

---

## 📋 Overview

Urban traffic congestion causes significant economic and environmental damage. This project implements **PressLight**, a deep reinforcement learning approach that learns to control traffic signals by minimizing traffic pressure—the imbalance between incoming and outgoing vehicles at intersections.

### Key Features

- ✅ **Production-Ready Implementation**: Fully documented PyTorch DQN agent with stability enhancements
- ✅ **Real-World Validation**: Tested on NYC 16×1 arterial network with realistic traffic patterns from New York City
- ✅ **Comprehensive Analysis**: 150 training episodes with rigorous hyperparameter tuning
- ✅ **Reproducible Results**: Docker containerization for one-command reproduction
- ✅ **Open Source**: Complete codebase with detailed implementation insights

### Performance Highlights

| Metric | Fixed-Time Baseline | PressLight Agent (150 ep) | Improvement |
|--------|---------------------|---------------------------|-------------|
| **Average Pressure** | 47.95 | 44.06 | **-8.10%** |
| **Cumulative Reward** | -172.63 | -158.64 | **-8.10%** |
| **Training Stability** | N/A | Stable convergence | ✓ |

> **Note**: Best results achieved with learning rate α=0.00005, slow epsilon decay, and 150+ episodes

---

## 🚀 Quick Start

### Option 1: Docker (Recommended - One Command)

```bash
# Builds image, trains agent (150 episodes), and evaluates
docker-compose up
```

**What happens:**
1. Builds Docker image with all dependencies
2. Starts container `rl_presslight_training_2`
3. Runs 150 episodes of training (~2 hours on CPU)
4. Evaluates trained model vs baseline
5. Saves all results to `./results/`

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/cityflow-project/CityFlow.git

# Quick test (50 episodes)
python -m src.run_experiment --episodes 50

# Recommended training (150 episodes)
python -m src.run_experiment --episodes 150
```

### Option 3: Custom Configuration

```bash
python -m src.run_experiment \
    --roadnet data/NewYork/roadnet_16_1.json \
    --flow data/NewYork/anon_16_1_300_newyork_real_1.json \
    --intersection intersection_1_1 \
    --episodes 150 \
    --lr 0.00005 \
    --gamma 0.99 \
    --batch_size 32
```

---

## 📁 Project Structure

```
RL_PROJECT_2/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── presslight.py          # PressLight DQN implementation
│   │   └── fixed_time.py          # Baseline controller
│   ├── environments/
│   │   ├── __init__.py
│   │   └── cityflow_env.py        # CityFlow wrapper with pressure rewards
│   ├── __init__.py
│   ├── run_experiment.py          # Main training script
│   └── evaluate_comparison.py     # Baseline comparison
├── data/
│   ├── NewYork/
│   │   ├── roadnet_16_1.json                     # Network topology
│   │   └── anon_16_1_300_newyork_real_1.json    # Real NYC traffic data
│                       
├── results/
│   ├── figures/                   # Training plots (PNG)
│   ├── logs/                      # Training metrics (CSV)
│   └── models/                    # Model checkpoints (.pth)
├── cityflow_config.json           # Simulator configuration
├── temp_cityflow_config.json      # Runtime config
├── docker-compose.yml             # One-command orchestration
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
└── Readme.md                      # This file
```

---

## 🔧 Methodology

### 1. Problem Formulation

**Traffic Pressure** measures vehicle imbalance at intersections:

```
P_t = |Σ(vehicles_incoming) - Σ(vehicles_outgoing)|
```

**Reward Function**: `r_t = -P_t / 100` (normalized for numerical stability)

**Goal**: Minimize pressure → Balance traffic flow → Reduce congestion

### 2. State Representation (56 dimensions)

```python
state = [
    phase_encoding,      # 8-dim one-hot (current traffic phase)
    incoming_lanes,      # 36-dim (12 lanes × 3 spatial segments)
    outgoing_lanes       # 12-dim (vehicle counts per lane)
]
```

**Critical Innovation**: Spatial segmentation divides each incoming lane into 3 segments (near/middle/far from intersection) to capture vehicle proximity and enable predictive control.

### 3. Network Architecture

```
Input(56) → Dense(128) → ReLU → Dense(64) → ReLU → Output(8)
```

- **Input**: 56-dimensional state vector
- **Hidden Layers**: [128, 64] neurons with ReLU activation
- **Output**: 8 Q-values (one per traffic phase)
- **Parameters**: ~11K trainable weights

### 4. Training Algorithm: DQN with Enhancements

**Core Components:**
- ✓ Experience Replay (buffer size: 10,000 transitions ≈ 28 episodes)
- ✓ Target Network (synchronized every 100 steps)
- ✓ Gradient Clipping (max norm = 1.0)
- ✓ Epsilon-Greedy Exploration (ε: 1.0 → 0.01)

**Loss Function:**
```
L(θ) = E[(r + γ·max_a' Q(s', a'; θ⁻) - Q(s, a; θ))²]
```

where γ = 0.99 (discount factor) and θ⁻ = target network parameters

---

## 🎯 Hyperparameters (Optimized Configuration)

| Parameter | Value | Critical Notes |
|-----------|-------|----------------|
| **Learning Rate (α)** | 0.00005 | **CRITICAL**: Values > 0.0001 cause training instability |
| **Discount Factor (γ)** | 0.99 | Long-term traffic flow optimization |
| **Batch Size** | 32 | Balance between variance and computation |
| **Replay Buffer** | 10,000 | ~28 episodes of experience |
| **Target Update Freq** | 100 steps | Prevents moving target problem |
| **Initial Epsilon (ε₀)** | 1.0 | Full exploration at start |
| **Epsilon Decay** | 0.97/episode | **Slow decay for better exploration** |
| **Min Epsilon (εₘᵢₙ)** | 0.01 | Maintain minimal exploration |
| **Optimizer** | Adam | Adaptive learning rates |
| **State Normalization** | ÷50.0 | **CRITICAL**: Prevents gradient explosion |
| **Reward Normalization** | ÷100 | Keeps rewards in [-2, 0] range |
| **Training Episodes** | 150 | Optimal balance (convergence vs time) |

---

## 📊 Experimental Results

### Baseline Comparison

The Fixed-Time Controller operates on a standard 90-second cycle (30s per phase × 3 main phases):
- **Average Pressure**: 47.95
- **Cumulative Reward**: -172.63

Our PressLight agent (150 episodes, α=0.00005) achieves:
- **Average Pressure**: 44.06
- **Cumulative Reward**: -158.64
- **Improvement**: **8.10% reduction in traffic pressure**

### Ablation Studies

#### Ablation 1: Epsilon Decay Speed (Exploration vs Exploitation)

| Configuration | Behavior | Outcome |
|---------------|----------|---------|
| **Fast Decay (0.97 with α=0.001)** | Quick stabilization by episode 25 | Risk of premature convergence; limited exploration |
| **Slow Decay (0.99 with α=0.001)** | Extended volatility up to episode 20 | Better loss reduction; suggests improved learning |

**Key Insight**: Slow epsilon decay allows more thorough exploration of the state space, leading to better final policies despite initial instability.

#### Ablation 2: Training Duration (Computational Budget)

| Episodes | Best Reward | Convergence Range (last 50 ep) | Notes |
|----------|-------------|-------------------------------|-------|
| 50 | ~-165 | High variance | Insufficient exploration |
| 150 | **-158.64** | -155 to -158 | **Optimal** |
| 300 | -152 | -152 to -157 | Marginal gains; 2× training time |

**Key Insight**: 150 episodes provides the best balance between performance and computational cost. Extended training (300 ep) yields only 3.7% additional improvement.

#### Ablation 3: Learning Rate Sensitivity

| Learning Rate | Episodes | Outcome |
|---------------|----------|---------|
| **0.001** | 150 | Training instability; divergence risk |
| **0.0001** | 150 | Underfitting; slow convergence with slow ε decay |
| **0.00005** | 150 | **Best results**: Stable convergence, reward ∈ [-155, -158] |

**Key Insight**: Learning rate is the most critical hyperparameter. The "sweet spot" at α=0.00005 enables stable learning while maintaining sufficient plasticity.

### Training Dynamics

**Convergence Pattern** (α=0.00005, 150 episodes):
- Episodes 1-30: High exploration; rewards range [-180, -165]
- Episodes 30-80: Gradual improvement; learning policy structure
- Episodes 80-150: **Stable convergence** in [-155, -158] range
- Loss function shows steady decrease with minimal perturbations

---

## 🔬 Key Findings & Implementation Insights

### Critical Success Factors

1. **State Normalization is Non-Negotiable**
   - Without dividing vehicle counts by 50.0, gradient explosion occurs within 10 episodes
   - Normalization keeps gradients bounded and enables stable backpropagation

2. **Learning Rate Selection is Critical**
   - Values above 0.0001 caused training instability and policy divergence
   - Conservative α=0.00005 trades training speed for reliability
   - Must be tuned jointly with epsilon decay rate

3. **Spatial Segmentation Provides Actionable Information**
   - 3-segment lane representation captures vehicle proximity
   - Near-intersection vehicles have immediate impact on pressure
   - Enables predictive rather than reactive control

4. **Pressure-Based Rewards Enable Interpretable Optimization**
   - Direct correlation between reward improvement and traffic flow
   - Clear optimization target (minimize vehicle imbalance)
   - Facilitates debugging and policy analysis

### Implementation Challenges & Solutions

| Challenge | Solution | Impact |
|-----------|----------|--------|
| **Gradient Explosion** | State normalization (÷50) + gradient clipping (max norm 1.0) | Prevents training failure |
| **Unstable Training** | Conservative learning rate (α=0.00005) | Stable convergence |
| **Premature Convergence** | Slow epsilon decay (0.97) with εₘᵢₙ=0.01 | Better exploration |
| **Sample Inefficiency** | Experience replay buffer (10K transitions) | Breaks temporal correlation |
| **Overfitting** | Modest network [128, 64] + target network | Generalizes well |

---

## 💻 Computational Requirements

- **Hardware**: Intel Core i9 CPU, 32GB RAM
  - GPU optional (CityFlow is CPU-bound)
- **Training Time**: 
  - 50 episodes: ~30 minutes
  - 150 episodes: ~2 hours ⭐ **Recommended**
  
- **Simulation Speed**: ~50 sim-seconds/real-second
- **Peak Memory**: 4GB RAM
- **Storage**: <500MB for all results

> **Note**: CityFlow's microscopic simulation is CPU-intensive. GPU acceleration provides minimal benefit for this application.

---

## 📈 Outputs & Results

After training, all results are saved to `./results/`:

```
results/
├── figures/
│   ├── training_curve_ep150.png      # Reward/loss over time
│   └── pressure_comparison.png       # Agent vs baseline
├── logs/
│   └── training_log_ep150.csv        # Episode-by-episode metrics
└── models/
    ├── presslight_ep50.pth                     # Checkpoints every 50 episodes
    ├── presslight_ep100.pth
    └── presslight_ep150.pth                    # Final model
```

### Interpreting Training Metrics

- **Cumulative Reward**: Higher (less negative) is better; target > -160
- **Average Pressure**: Lower is better; indicates balanced traffic flow
- **Loss**: Should decrease and stabilize; spikes indicate learning
- **Epsilon**: Should decay from 1.0 → 0.01 over training

---

## 🐳 Docker Usage Guide

### Quick Commands

```bash
# One-command execution (build + train + evaluate)
docker-compose up

# Build image only
docker-compose build

# Training only (150 episodes)
docker-compose run rl-training python -m src.run_experiment --episodes 150

# Evaluation only (using saved model)
docker-compose run rl-training python -m src.evaluate_comparison \
    --model_path results/models/presslight_ep150.pth

# Interactive shell (for debugging)
docker-compose run rl-training bash

# View logs in real-time
docker-compose logs -f
```

### Dockerfile Architecture

```dockerfile
FROM python:3.9-slim                          # Lightweight Python base
RUN apt-get update && apt-get install ...     # System dependencies (gcc, git)
COPY requirements.txt .                       # Python package list
RUN pip install -r requirements.txt           # Install PyTorch, numpy, etc.
RUN pip install git+https://...CityFlow.git   # Install traffic simulator
COPY . /app                                   # Copy project files
WORKDIR /app                                  # Set working directory
RUN mkdir -p results/{logs,models,figures}    # Create output directories
CMD ["sh", "-c", "python -m src.run_experiment --episodes 150 && python -m src.evaluate_comparison --model_path results/models/presslight_ep150.pth"]
```

### Volume Mounting

Docker automatically mounts local directories for persistent storage:
- `./results` → `/app/results` (training outputs)
- `./data` → `/app/data` (traffic network data)

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove all unused Docker resources
docker system prune -a

# Remove specific image
docker rmi presslight-rl
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Stochastic Instability**: Late-stage reward curves show variability, indicating the agent hasn't reached safety-critical reliability levels required for real-world deployment

2. **Perfect State Assumption**: Model assumes noise-free sensor observations, ignoring real-world sensor errors inherent in Intelligent Transportation Systems (ITS)

3. **Single-Intersection Optimization**: Agent operates locally without global coordination necessary for city-scale traffic management

4. **Deterministic Environment**: CityFlow provides reproducible conditions; real traffic has unpredictable human behavior

5. **Computational Cost**: Extended training (150+ episodes) required for optimal performance

### Future Research Directions

- **Multi-Agent Coordination**: Extend to coordinate multiple intersections simultaneously
- **Robust Policy Learning**: Incorporate domain randomization for sensor noise
- **Transfer Learning**: Pre-train on diverse networks for faster adaptation
- **Safety Constraints**: Add explicit safety guarantees (e.g., maximum wait times)
- **Real-World Validation**: Deploy in hardware-in-the-loop testbeds

---

## 📚 References

1. **Wei, H., Zheng, G., Yao, H., & Li, Z.** (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*. KDD 2019.

2. **Mnih, V. et al.** (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.

3. **Zhang, H. et al.** (2019). *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario*. WWW 2019.

---

## 👥 Authors

**Sami BACHTAOUI** - bachtaouisami@gmail.com  
**Ayoub EL ASSIOUI**

*Department of Computer Science*  
*Reinforcement Learning Project, Rabat, Morocco, 2025*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- CityFlow team for the microscopic traffic simulation platform
- PressLight authors (Wei et al., 2019) for the pressure-based control framework
- New York City DOT for real-world traffic flow data

---

## 🔗 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{bachtaoui2025presslight,
  title={Deep Reinforcement Learning for Adaptive Traffic Signal Control: 
         A PressLight Implementation Study},
  author={Bachtaoui, Sami and El Assioui, Ayoub},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/presslight-traffic-control}}
}
```

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Training crashes with gradient explosion  
**Solution**: Ensure state normalization is enabled (÷50.0 in config)

**Issue**: Slow convergence after 50 episodes  
**Solution**: Reduce learning rate to 0.00005 and extend training to 150 episodes

**Issue**: Docker container exits immediately  
**Solution**: Check logs with `docker-compose logs` for dependency errors

**Issue**: CityFlow installation fails  
**Solution**: Install build-essential: `apt-get install build-essential`

---

**Repository**: https://github.com/samisb8/PressLight
**Last Updated**: January 2026
**Version**: 1.0.0
