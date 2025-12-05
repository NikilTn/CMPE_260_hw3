from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlgorithmConfig:
    algo_name: str = "reinforce"
    gamma: float = 0.99
    episodes: int = 1000
    max_steps_per_episode: int = 500
    policy_lr: float = 1e-3
    value_lr: float = 5e-4
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    normalize_advantages: bool = True
    entropy_coef: float = 0.01
    n_step: int = 1
    seed: int = 42
    log_interval: int = 10
    save_dir: str = "results"
    solved_threshold: float = 475.0
    solved_window: int = 100
    track_gradient_norm: bool = False


@dataclass
class ExperimentConfig:
    algorithms: List[str] = field(default_factory=lambda: [
        'reinforce', 'reinforce_baseline', 'actor_critic'
    ])
    num_seeds: int = 5
    base_config: Optional[AlgorithmConfig] = None
    plot_confidence_interval: bool = True
    save_plots: bool = True
    track_gradient_norm: bool = False


def get_config(algo_name, **kwargs):
    config = AlgorithmConfig(algo_name=algo_name)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


