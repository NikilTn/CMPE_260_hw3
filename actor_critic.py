import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional
from models import PolicyNetwork, ValueNetwork
from utils import normalize_advantages, compute_gradient_norm, compute_n_step_return
from config import AlgorithmConfig


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_dim, action_dim, config.hidden_layers).to(self.device)
        self.value_net = ValueNetwork(state_dim, config.hidden_layers).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.value_lr)
        self.reset_episode_buffer()
    
    def reset_episode_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.select_action(state_tensor)
        value = self.value_net(state_tensor)
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.squeeze().item())
        self.dones.append(done)
    
    def update_step(self,
                    state: torch.Tensor,
                    log_prob: torch.Tensor,
                    reward: float,
                    next_state: torch.Tensor,
                    done: bool,
                    value: torch.Tensor) -> Tuple[float, float, Optional[float]]:
        with torch.no_grad():
            if done:
                next_value = 0.0
            else:
                next_value = self.value_net(next_state).squeeze().item()
            
            td_target = reward + self.config.gamma * next_value
            
            advantage = td_target - value.squeeze().item()
        
        td_target_tensor = torch.tensor(td_target, dtype=torch.float32).to(self.device)
        
        critic_loss = nn.SmoothL1Loss()(value.squeeze(), td_target_tensor)
        
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32).to(self.device)
        actor_loss = -(log_prob * advantage_tensor)
        
        if self.config.entropy_coef > 0:
            dist = self.policy.get_action_distribution(state)
            entropy = dist.entropy()
            actor_loss = actor_loss - self.config.entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = None
        if self.config.track_gradient_norm:
            grad_norm = compute_gradient_norm(self.policy)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), grad_norm
    
    def update_episode(self) -> Tuple[float, float, Optional[float]]:
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        log_probs_tensor = torch.stack(self.log_probs).to(self.device)
        
        if self.config.n_step > 1:
            returns = compute_n_step_return(
                self.rewards, self.values, self.config.gamma,
                self.config.n_step, done=self.dones[-1]
            )
        else:
            returns = []
            for t in range(len(self.rewards)):
                if t == len(self.rewards) - 1:
                    td_target = self.rewards[t] if self.dones[-1] else \
                               self.rewards[t] + self.config.gamma * self.values[t]
                else:
                    td_target = self.rewards[t] + self.config.gamma * self.values[t + 1]
                returns.append(td_target)
        
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(self.values).to(self.device)
        advantages = returns_tensor - values_tensor
        
        if self.config.normalize_advantages:
            advantages = torch.FloatTensor(
                normalize_advantages(advantages.cpu().numpy())
            ).to(self.device)
        
        predicted_values = self.value_net(states_tensor).squeeze()
        critic_loss = nn.MSELoss()(predicted_values, returns_tensor)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        if self.config.entropy_coef > 0:
            logits = self.policy(states_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean()
            actor_loss = actor_loss - self.config.entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = None
        if self.config.track_gradient_norm:
            grad_norm = compute_gradient_norm(self.policy)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        self.reset_episode_buffer()
        return actor_loss.item(), critic_loss.item(), grad_norm


def train_actor_critic(config, online_update=True, return_gradients: bool = False):
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCriticAgent(state_dim, action_dim, config)
    episode_returns = []
    gradient_norms = [] if config.track_gradient_norm or return_gradients else None
    
    if not hasattr(config, 'track_gradient_norm'):
        config.track_gradient_norm = False
    
    update_mode = "online (TD(0))" if online_update else f"episodic (n={config.n_step})"
    print(f"Training Actor-Critic with {update_mode}...")
    print(f"γ={config.gamma}, lr={config.policy_lr}, episodes={config.episodes}")
    
    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        ep_return = 0.0
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        ep_grad_norms = [] if gradient_norms is not None else None
        num_steps = 0
        
        for step in range(config.max_steps_per_episode):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if online_update:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                actor_loss, critic_loss, grad_norm = agent.update_step(
                    state_tensor, log_prob, reward, next_state_tensor, done, value
                )
                ep_actor_loss += actor_loss
                ep_critic_loss += critic_loss
                if ep_grad_norms is not None and grad_norm is not None:
                    ep_grad_norms.append(grad_norm)
            else:
                agent.store_transition(state, action, log_prob, reward, value, done)
            
            ep_return += reward
            num_steps += 1
            state = next_state
            if done:
                break
        
        if not online_update:
            actor_loss, critic_loss, grad_norm = agent.update_episode()
            ep_actor_loss = actor_loss
            ep_critic_loss = critic_loss
            if ep_grad_norms is not None and grad_norm is not None:
                ep_grad_norms.append(grad_norm)
        else:
            if num_steps > 0:
                ep_actor_loss /= num_steps
                ep_critic_loss /= num_steps
        
        if gradient_norms is not None:
            mean_grad = float(np.mean(ep_grad_norms)) if ep_grad_norms else None
            gradient_norms.append(mean_grad)
        
        episode_returns.append(ep_return)
        
        if episode % config.log_interval == 0:
            avg = np.mean(episode_returns[-100:])
            print(f"Episode {episode}/{config.episodes} | Return: {ep_return:.2f} | "
                  f"Avg(100): {avg:.2f} | Actor Loss: {ep_actor_loss:.4f} | "
                  f"Critic Loss: {ep_critic_loss:.4f}")
        
        if len(episode_returns) >= config.solved_window:
            if np.mean(episode_returns[-config.solved_window:]) >= config.solved_threshold:
                print(f"\nSolved in {episode} episodes!")
                break
    
    env.close()
    if return_gradients:
        return episode_returns, gradient_norms
    return episode_returns


if __name__ == "__main__":
    from config import get_config
    config = get_config("actor_critic", episodes=500)
    returns = train_actor_critic(config, online_update=True)
    print(f"\nFinal average return (last 100): {np.mean(returns[-100:]):.2f}")

