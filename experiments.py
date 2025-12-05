import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import Dict, List
import csv

from config import get_config, ExperimentConfig, AlgorithmConfig
from utils import (set_seed, create_results_directory, 
                   plot_learning_curve_with_confidence, moving_average)
from reinforce import train_reinforce
from actor_critic import train_actor_critic


def run_single_experiment(algo: str, seed: int, episodes: int = 1000) -> List[float]:
    """
    Run a single experiment for a given algorithm and seed.
    
    Args:
        algo: Algorithm name
        seed: Random seed
        episodes: Number of episodes to train
        
    Returns:
        List of episode returns
    """
    # Create config
    config = get_config(
        algo_name=algo,
        episodes=episodes,
        seed=seed,
        log_interval=50  # Less frequent logging for experiments
    )
    
    # Set seed
    env = gym.make('CartPole-v1')
    set_seed(seed, env)
    env.close()
    
    # Train
    print(f"Running {algo} with seed {seed}...")
    
    if algo == 'reinforce':
        returns = train_reinforce(config, use_baseline=False)
    elif algo == 'reinforce_baseline':
        returns = train_reinforce(config, use_baseline=True)
    elif algo == 'actor_critic':
        returns = train_actor_critic(config, online_update=True)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return returns


def run_compare_algorithms(exp_config: ExperimentConfig = None):
    """
    Compare multiple algorithms across multiple seeds.
    
    This is the main ablation study function that:
    1. Runs each algorithm for multiple random seeds
    2. Computes mean and std of returns across seeds
    3. Plots learning curves with confidence intervals
    4. Saves results and statistics
    
    Args:
        exp_config: Experiment configuration
    """
    if exp_config is None:
        base_config = AlgorithmConfig()
        exp_config = ExperimentConfig(base_config=base_config)
    
    # Create results directory
    save_dir = exp_config.base_config.save_dir
    create_results_directory(save_dir)
    
    print("\n" + "="*60)
    print("ABLATION STUDY: Comparing Policy Gradient Methods")
    print("="*60)
    print(f"Algorithms: {exp_config.algorithms}")
    print(f"Seeds: {exp_config.num_seeds}")
    print(f"Episodes per run: {exp_config.base_config.episodes}")
    print("="*60 + "\n")
    
    # Store results: {algo_name: [run1_returns, run2_returns, ...]}
    all_results = {}
    
    # Run experiments
    for algo in exp_config.algorithms:
        print(f"\n--- Running {algo.upper()} ---")
        algo_results = []
        
        for seed_idx in range(exp_config.num_seeds):
            seed = exp_config.base_config.seed + seed_idx
            returns = run_single_experiment(
                algo, 
                seed, 
                exp_config.base_config.episodes
            )
            algo_results.append(returns)
            
            # Save individual run
            filename = f"{save_dir}/{algo}_seed{seed}.csv"
            save_individual_run(returns, filename, algo, seed)
        
        all_results[algo] = algo_results
    
    # Compute and print statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    stats = {}
    for algo, runs in all_results.items():
        # Convert to numpy array: (num_seeds, num_episodes)
        runs_array = np.array(runs)
        
        # Final performance (average of last 100 episodes)
        final_returns = []
        for run in runs:
            final_returns.append(np.mean(run[-100:]))
        
        mean_final = np.mean(final_returns)
        std_final = np.std(final_returns)
        
        # Sample efficiency: episodes to reach threshold (e.g., 195)
        threshold = 195.0
        episodes_to_threshold = []
        for run in runs:
            smoothed = moving_average(run, window=100)
            solved_at = np.where(smoothed >= threshold)[0]
            if len(solved_at) > 0:
                episodes_to_threshold.append(solved_at[0] + 100)  # +100 for window
            else:
                episodes_to_threshold.append(len(run))  # Didn't solve
        
        mean_episodes = np.mean(episodes_to_threshold)
        std_episodes = np.std(episodes_to_threshold)
        
        stats[algo] = {
            'final_return_mean': mean_final,
            'final_return_std': std_final,
            'episodes_to_threshold_mean': mean_episodes,
            'episodes_to_threshold_std': std_episodes
        }
        
        print(f"\n{algo.upper()}:")
        print(f"  Final Return (last 100): {mean_final:.2f} ± {std_final:.2f}")
        print(f"  Episodes to reach {threshold}: {mean_episodes:.1f} ± {std_episodes:.1f}")
    
    print("\n" + "="*60 + "\n")
    
    # Save statistics
    stats_file = f"{save_dir}/comparison_statistics.csv"
    save_comparison_stats(stats, stats_file)
    
    # Plot comparison with confidence intervals
    if exp_config.save_plots:
        plot_path = f"{save_dir}/comparison_all_algorithms.png"
        plot_learning_curve_with_confidence(
            all_results,
            title="Policy Gradient Methods Comparison on CartPole-v1",
            xlabel="Episode",
            ylabel="Return",
            save_path=plot_path,
            window=50
        )
        
        print(f"Comparison plot saved to {plot_path}")
    
    return all_results, stats


def save_individual_run(returns: List[float], 
                       filepath: str,
                       algorithm: str,
                       seed: int):
    """Save individual run results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Return', 'Algorithm', 'Seed'])
        for episode, ret in enumerate(returns, 1):
            writer.writerow([episode, ret, algorithm, seed])


def save_comparison_stats(stats: Dict, filepath: str):
    """Save comparison statistics to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 
            'Final_Return_Mean', 
            'Final_Return_Std',
            'Episodes_To_195_Mean',
            'Episodes_To_195_Std'
        ])
        
        for algo, stat_dict in stats.items():
            writer.writerow([
                algo,
                f"{stat_dict['final_return_mean']:.2f}",
                f"{stat_dict['final_return_std']:.2f}",
                f"{stat_dict['episodes_to_threshold_mean']:.1f}",
                f"{stat_dict['episodes_to_threshold_std']:.1f}"
            ])
    
    print(f"Statistics saved to {filepath}")


def run_gradient_variance_analysis(episodes: int = 500, num_seeds: int = 3):
    """
    Analyze and compare gradient variance between algorithms.
    
    Tracks the L2 norm of policy gradients during training
    to empirically verify variance reduction from bootstrapping/baselines.
    
    Args:
        episodes: Number of episodes to train
        num_seeds: Number of random seeds
    """
    print("\n" + "="*60)
    print("GRADIENT VARIANCE ANALYSIS")
    print("="*60 + "\n")
    
    algorithms = ['reinforce', 'actor_critic']
    save_dir = "results/gradient_analysis"
    create_results_directory(save_dir)
    
    grad_results = {}
    
    for algo in algorithms:
        algo_grad_runs = []
        print(f"\n--- Tracking gradients for {algo.upper()} ---")
        for seed_idx in range(num_seeds):
            seed = 42 + seed_idx
            config = get_config(
                algo_name=algo,
                episodes=episodes,
                seed=seed,
                log_interval=episodes + 1,  # silence logs
                track_gradient_norm=True,
                save_dir=save_dir
            )
            
            env = gym.make('CartPole-v1')
            set_seed(seed, env)
            env.close()
            
            if algo == 'reinforce':
                _, grad_norms = train_reinforce(
                    config, use_baseline=False, return_gradients=True
                )
            else:
                _, grad_norms = train_actor_critic(
                    config, online_update=True, return_gradients=True
                )
            
            if grad_norms is None:
                grad_norms = []
            algo_grad_runs.append(grad_norms)
            
            grad_file = f"{save_dir}/{algo}_grad_seed{seed}.csv"
            save_grad_norms(grad_norms, grad_file, algo, seed)
        
        grad_results[algo] = algo_grad_runs
    
    variance_plot_path = f"{save_dir}/gradient_variance.png"
    plot_gradient_variance(grad_results, save_path=variance_plot_path)
    print(f"Gradient variance plot saved to {variance_plot_path}")
    
    return grad_results


def save_grad_norms(grad_norms: List[float], filepath: str,
                   algorithm: str, seed: int):
    """Persist gradient norms for later analysis."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'GradNorm', 'Algorithm', 'Seed'])
        for ep, g in enumerate(grad_norms, 1):
            writer.writerow([ep, g, algorithm, seed])


def plot_gradient_variance(grad_results: Dict[str, List[List[float]]],
                          save_path: str = "results/gradient_analysis/gradient_variance.png"):
    plt.figure(figsize=(10, 6))
    
    for algo, runs in grad_results.items():
        valid_runs = [np.array(r, dtype=np.float32) for r in runs if r is not None and len(r) > 0]
        if len(valid_runs) == 0:
            continue
        min_len = min(len(r) for r in valid_runs)
        stacked = np.stack([r[:min_len] for r in valid_runs], axis=0)
        variance = np.var(stacked, axis=0)
        episodes = np.arange(1, min_len + 1)
        plt.plot(episodes, variance, label=algo)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Variance of Gradient Norm', fontsize=12)
    plt.title('Policy Gradient Variance Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_efficiency(all_results: Dict[str, List[List[float]]], 
                          save_path: str = "results/sample_efficiency.png"):
    """
    Plot sample efficiency: episodes needed to reach different return thresholds.
    
    Args:
        all_results: Dictionary mapping algorithm names to list of runs
        save_path: Path to save plot
    """
    thresholds = [150, 195, 300, 400, 475]
    
    plt.figure(figsize=(10, 6))
    
    for algo, runs in all_results.items():
        episodes_needed = {th: [] for th in thresholds}
        
        for run in runs:
            smoothed = moving_average(run, window=100)
            for threshold in thresholds:
                solved_at = np.where(smoothed >= threshold)[0]
                if len(solved_at) > 0:
                    episodes_needed[threshold].append(solved_at[0] + 100)
                else:
                    episodes_needed[threshold].append(len(run))
        
        means = [np.mean(episodes_needed[th]) for th in thresholds]
        stds = [np.std(episodes_needed[th]) for th in thresholds]
        
        plt.errorbar(thresholds, means, yerr=stds, marker='o', 
                    label=algo, capsize=5, linewidth=2, markersize=8)
    
    plt.xlabel('Return Threshold', fontsize=12)
    plt.ylabel('Episodes Needed', fontsize=12)
    plt.title('Sample Efficiency Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample efficiency plot saved to {save_path}")
    plt.close()


def main():
    """Main function to run ablation studies."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per run')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['reinforce', 'reinforce_baseline', 'actor_critic'],
                       help='Algorithms to compare')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create experiment config
    from config import AlgorithmConfig
    base_config = AlgorithmConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        seed=42
    )
    
    exp_config = ExperimentConfig(
        algorithms=args.algorithms,
        num_seeds=args.num_seeds,
        base_config=base_config
    )
    
    # Run comparison
    all_results, stats = run_compare_algorithms(exp_config)
    
    # Plot sample efficiency
    plot_sample_efficiency(all_results, 
                          save_path=f"{args.save_dir}/sample_efficiency.png")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"All results saved to {args.save_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


