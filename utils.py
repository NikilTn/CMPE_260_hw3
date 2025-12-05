import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Optional
import gymnasium as gym


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env:
        env.reset(seed=seed)


def discount_cumsum(rewards, gamma):
    returns = []
    running_return = 0.0
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    return np.array(returns, dtype=np.float32)


def compute_n_step_return(rewards, values, gamma, n=1, done=True):
    T = len(rewards)
    returns = []
    for t in range(T):
        ret = 0.0
        for k in range(min(n, T - t)):
            ret += (gamma ** k) * rewards[t + k]
        if t + n < T:
            ret += (gamma ** n) * values[t + n]
        returns.append(ret)
    return returns


def moving_average(data, window):
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curve(data_dict, title="Learning Curve", xlabel="Episode", 
                        ylabel="Return", save_path=None, window=50, show_raw=False):
    plt.figure(figsize=(10, 6))
    
    for algo_name, returns in data_dict.items():
        episodes = np.arange(1, len(returns) + 1)
        
        if show_raw:
            plt.plot(episodes, returns, alpha=0.3, linewidth=0.5)
        
        # Plot smoothed curve
        if len(returns) >= window:
            smoothed = moving_average(returns, window)
            smoothed_episodes = episodes[window-1:]
            plt.plot(smoothed_episodes, smoothed, label=algo_name, linewidth=2)
        else:
            plt.plot(episodes, returns, label=algo_name, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_learning_curve_with_confidence(data_dict, title="Learning Curve",
                                        xlabel="Episode", ylabel="Return",
                                        save_path=None, window=50):
    plt.figure(figsize=(10, 6))
    
    for algo_name, runs in data_dict.items():
        runs_array = np.array(runs)
        mean_returns = np.mean(runs_array, axis=0)
        std_returns = np.std(runs_array, axis=0)
        episodes = np.arange(1, len(mean_returns) + 1)
        
        if window > 1:
            mean_returns = moving_average(mean_returns, window)
            std_returns = moving_average(std_returns, window)
            episodes = episodes[window-1:]
        
        plt.plot(episodes, mean_returns, label=algo_name, linewidth=2)
        plt.fill_between(episodes, mean_returns - std_returns, 
                        mean_returns + std_returns, alpha=0.2)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def normalize_advantages(advantages, eps=1e-8):
    if len(advantages) <= 1:
        return advantages
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < eps:
        return advantages - mean
    return (advantages - mean) / (std + eps)


def create_results_directory(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_results(returns, filepath, algorithm, seed):
    import csv
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Return', 'Algorithm', 'Seed'])
        for ep, ret in enumerate(returns, 1):
            writer.writerow([ep, ret, algorithm, seed])
    print(f"Saved to {filepath}")


