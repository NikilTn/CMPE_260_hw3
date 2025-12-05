import torch
import torch.nn as nn
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        super(PolicyNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)
    
    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: List[int]):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        super(ActorCriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
        )
    
    def forward(self, state):
        shared = self.shared(state)
        return self.policy_head(shared), self.value_head(shared)


