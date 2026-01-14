import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional
from pathlib import Path


#Detect if there is a gpu and use it 
device = torch.device(
    "cuda" if torch.cuda.is_available() #NVIDIA GPU
    else "mps" if torch.backends.mps.is_available() # Apple Silicon GPU
    else "cpu" 
)


class PressLightNet(nn.Module):
    """
    Deep Q-Network for PressLight algorithm.
    
    Architecture described in Section 4.2 of the paper.
    Takes as input the state vector containing:
    - Current phase (one-hot encoded)
    - Vehicle counts on incoming lanes (segmented)
    - Vehicle counts on outgoing lanes
    
    Outputs Q-values for each possible traffic signal phase.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [128, 64]):
        """
        Initialize the DQN network.
        
        Args:
            state_dim: Dimension of the state vector
            action_dim: Number of possible phases (actions)
            hidden_sizes: List of hidden layer sizes for the MLP
        """
        super(PressLightNet, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build MLP layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores transitions and samples mini-batches for training.
    Uses a circular buffer (deque) for memory efficiency.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity) #how many last moves the agent remember.
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                                 torch.Tensor, torch.Tensor, 
                                                 torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)


class PressLightAgent:
    """
    PressLight Agent implementing DQN-based traffic signal control.
    
    Implements the learning algorithm described in Section 4 of the paper.
    Uses Double DQN with target network for stable training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        hidden_sizes: List[int] = [128, 64]
    ):
        """
        Initialize the PressLight agent.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible phases
            lr: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            target_update_freq: Frequency (in updates) to sync target network
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for training
            hidden_sizes: Hidden layer sizes for the DQN
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.policy_net = PressLightNet(state_dim, action_dim, hidden_sizes).to(device) #created a Policy Net (the one that is always learning) and a Target Net (a "frozen" copy)
        self.target_net = PressLightNet(state_dim, action_dim, hidden_sizes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training step counter
        self.update_counter = 0
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action (phase index)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def update(self) -> Optional[float]:
        """
        Perform one step of DQN training.
        
        Implements the Q-learning update rule from Section 4.3:
        Loss = MSE(Q(s,a), r + γ * max_a' Q_target(s', a'))
        
        Returns:
            Training loss value, or None if buffer doesn't have enough samples
        """
        if len(self.memory) < self.batch_size: # You don't start learning until you have at least one batch_size
            return None
        
        # Sample mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss (MSE)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, path: str) -> None:
        """
        Save the agent's policy network.
        
        Args:
            path: File path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, path)
        
    def load_model(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: File path to load the model from
        """
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']


# Example usage
if __name__ == "__main__":
    # Example: Single intersection with 8 phases
    # State might include: 8 (one-hot phase) + 12 (incoming lanes) + 12 (outgoing lanes) = 32
    state_dim = 32
    action_dim = 8
    
    agent = PressLightAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        target_update_freq=100
    )
    
    print(f"PressLight agent initialized on device: {device}")
    print(f"Policy network: {agent.policy_net}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Simulate one step
    dummy_state = np.random.randn(state_dim)
    action = agent.get_action(dummy_state)
    print(f"Selected action: {action}")
    
    # Store transition
    next_state = np.random.randn(state_dim)
    agent.memory.push(dummy_state, action, 1.0, next_state, False)
    
    # Update (will return None since buffer is too small)
    loss = agent.update()
    print(f"Update loss: {loss}")