import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional
from models import PolicyNetwork, ValueNetwork
from utils import discount_cumsum, normalize_advantages, compute_gradient_norm
from config import AlgorithmConfig


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, config, use_baseline=False):
        self.config = config
        self.use_baseline = use_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_dim, action_dim, config.hidden_layers).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        
        if use_baseline:
            self.value_net = ValueNetwork(state_dim, config.hidden_layers).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.value_lr)
        else:
            self.value_net = None
        
        self.reset_episode_buffer()
        
    def reset_episode_buffer(self):
        self.states = []
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.select_action(state_tensor)
        return action, log_prob
    
    def store_transition(self, state, log_prob, reward):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def finish_episode(self):
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        log_probs_tensor = torch.stack(self.log_probs).to(self.device)
        returns = discount_cumsum(self.rewards, self.config.gamma)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        if self.use_baseline:
            with torch.no_grad():
                values = self.value_net(states_tensor).squeeze()
            advantages = returns_tensor - values
            value_loss = self._update_value_network(states_tensor, returns_tensor)
        else:
            advantages = returns_tensor
            value_loss = None
        
        if self.config.normalize_advantages:
            advantages = torch.FloatTensor(
                normalize_advantages(advantages.cpu().numpy())
            ).to(self.device)
        
        policy_loss = -(log_probs_tensor * advantages).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = None
        if self.config.track_gradient_norm:
            grad_norm = compute_gradient_norm(self.policy)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        self.reset_episode_buffer()
        return policy_loss.item(), value_loss, grad_norm
    
    def _update_value_network(self, states, returns):
        values = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        return value_loss.item()


def train_reinforce(config, use_baseline=False, return_gradients: bool = False):
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim, action_dim, config, use_baseline)
    episode_returns = []
    gradient_norms = [] if config.track_gradient_norm or return_gradients else None
    
    algo_name = 'REINFORCE with baseline' if use_baseline else 'REINFORCE'
    print(f"Training {algo_name}...")
    print(f"γ={config.gamma}, lr={config.policy_lr}, episodes={config.episodes}")
    
    for ep in range(1, config.episodes + 1):
        state, _ = env.reset()
        ep_return = 0.0
        
        for step in range(config.max_steps_per_episode):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, log_prob, reward)
            ep_return += reward
            state = next_state
            if done:
                break
        
        policy_loss, value_loss, grad_norm = agent.finish_episode()
        episode_returns.append(ep_return)
        if gradient_norms is not None:
            gradient_norms.append(grad_norm)
        
        if ep % config.log_interval == 0:
            avg = np.mean(episode_returns[-100:])
            msg = f"Episode {ep}/{config.episodes} | Return: {ep_return:.2f} | Avg(100): {avg:.2f} | Policy Loss: {policy_loss:.4f}"
            if value_loss:
                msg += f" | Value Loss: {value_loss:.4f}"
            print(msg)
        
        if len(episode_returns) >= config.solved_window:
            if np.mean(episode_returns[-config.solved_window:]) >= config.solved_threshold:
                print(f"\nSolved in {ep} episodes!")
                break
    
    env.close()
    if return_gradients:
        return episode_returns, gradient_norms
    return episode_returns


if __name__ == "__main__":
    # Simple test
    from config import get_config
    
    # Test REINFORCE without baseline
    config = get_config("reinforce", episodes=500)
    returns = train_reinforce(config, use_baseline=False)
    
    print(f"\nFinal average return (last 100): {np.mean(returns[-100:]):.2f}")

