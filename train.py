import argparse
import os
import numpy as np
import gymnasium as gym

from config import get_config, AlgorithmConfig
from utils import set_seed, create_results_directory, save_results, plot_learning_curve
from reinforce import train_reinforce
from actor_critic import train_actor_critic


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train policy gradient methods on CartPole-v1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm selection
    parser.add_argument('--algo', type=str, default='reinforce',
                       choices=['reinforce', 'reinforce_baseline', 'actor_critic'],
                       help='Algorithm to use')
    
    # Training hyperparameters
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for policy network')
    parser.add_argument('--value_lr', type=float, default=1e-3,
                       help='Learning rate for value network')
    
    # Network architecture
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer sizes')
    
    # Algorithm-specific
    parser.add_argument('--normalize_advantages', action='store_true', default=True,
                       help='Normalize advantages')
    parser.add_argument('--no_normalize_advantages', dest='normalize_advantages',
                       action='store_false',
                       help='Do not normalize advantages')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient for Actor-Critic')
    parser.add_argument('--n_step', type=int, default=1,
                       help='n-step for TD in Actor-Critic')
    parser.add_argument('--online_update', action='store_true', default=True,
                       help='Use online updates (TD(0)) for Actor-Critic')
    parser.add_argument('--episodic_update', dest='online_update',
                       action='store_false',
                       help='Use episodic updates for Actor-Critic')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (episodes)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--save_plot', action='store_true', default=True,
                       help='Save learning curve plot')
    parser.add_argument('--track_gradient_norm', action='store_true', default=False,
                       help='Track gradient norms during training')
    
    # Solving criteria
    parser.add_argument('--solved_threshold', type=float, default=475.0,
                       help='Threshold for considering environment solved')
    parser.add_argument('--solved_window', type=int, default=100,
                       help='Window for computing average return')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from arguments
    config = get_config(
        algo_name=args.algo,
        episodes=args.episodes,
        gamma=args.gamma,
        policy_lr=args.lr,
        value_lr=args.value_lr,
        hidden_layers=args.hidden_layers,
        normalize_advantages=args.normalize_advantages,
        entropy_coef=args.entropy_coef,
        n_step=args.n_step,
        seed=args.seed,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        solved_threshold=args.solved_threshold,
        solved_window=args.solved_window,
        track_gradient_norm=args.track_gradient_norm
    )
    
    # Create results directory
    create_results_directory(config.save_dir)
    
    # Set seed
    print(f"\n{'='*60}")
    print(f"Training {args.algo.upper()} on CartPole-v1")
    print(f"{'='*60}")
    print(f"Seed: {config.seed}")
    print(f"Episodes: {config.episodes}")
    print(f"Gamma: {config.gamma}")
    print(f"Policy LR: {config.policy_lr}")
    if args.algo in ['reinforce_baseline', 'actor_critic']:
        print(f"Value LR: {config.value_lr}")
    if args.algo == 'actor_critic':
        print(f"Entropy Coef: {config.entropy_coef}")
        print(f"n-step: {config.n_step}")
        print(f"Update Mode: {'Online' if args.online_update else 'Episodic'}")
    print(f"Hidden Layers: {config.hidden_layers}")
    print(f"{'='*60}\n")
    
    # Set seed for environment
    env = gym.make('CartPole-v1')
    set_seed(config.seed, env)
    env.close()
    
    # Train based on algorithm
    if args.algo == 'reinforce':
        episode_returns = train_reinforce(config, use_baseline=False)
    elif args.algo == 'reinforce_baseline':
        episode_returns = train_reinforce(config, use_baseline=True)
    elif args.algo == 'actor_critic':
        episode_returns = train_actor_critic(config, online_update=args.online_update)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episode_returns)}")
    print(f"Final return: {episode_returns[-1]:.2f}")
    print(f"Average return (last 100): {np.mean(episode_returns[-100:]):.2f}")
    print(f"Max return: {np.max(episode_returns):.2f}")
    print(f"Min return: {np.min(episode_returns):.2f}")
    
    # Check if solved
    if len(episode_returns) >= config.solved_window:
        avg_return = np.mean(episode_returns[-config.solved_window:])
        if avg_return >= config.solved_threshold:
            print(f"\n✓ Environment SOLVED! Average return: {avg_return:.2f}")
        else:
            print(f"\n✗ Not quite solved. Average return: {avg_return:.2f}")
    print(f"{'='*60}\n")
    
    # Save results
    result_filename = f"{config.save_dir}/{args.algo}_seed{config.seed}.csv"
    save_results(episode_returns, result_filename, args.algo, config.seed)
    
    # Plot learning curve
    if args.save_plot:
        plot_path = f"{config.save_dir}/{args.algo}_seed{config.seed}.png"
        plot_learning_curve(
            {args.algo: episode_returns},
            title=f"{args.algo.upper()} on CartPole-v1 (seed={config.seed})",
            save_path=plot_path,
            window=50,
            show_raw=True
        )
    
    print(f"Results saved to {config.save_dir}/")


if __name__ == "__main__":
    main()


