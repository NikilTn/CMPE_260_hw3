[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thalapaneninikil/rl_hw3/blob/main/run_project.ipynb)

# Policy Gradient Methods - CartPole-v1

 Three algorithms: REINFORCE, REINFORCE+Baseline, and Actor-Critic.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Test
python test_quick.py

# Train
python train.py --algo reinforce --episodes 500 --seed 42
python train.py --algo actor_critic --episodes 500 --seed 42

# Compare all algorithms
python experiments.py --num_seeds 5 --episodes 800
```

---


## Algorithms

### 1. REINFORCE
- Monte Carlo policy gradient
- High variance, no bias
- Updates at episode end

### 2. REINFORCE with Baseline
- Uses value function as baseline
- Reduces variance
- Still unbiased

### 3. Actor-Critic
- TD learning with bootstrapping
- Lowest variance
- Can update each step
- Some bias

---

## Training Commands

### Basic Usage

```bash
python train.py --algo ALGORITHM --episodes NUM --seed SEED
```

### Examples

```bash
# REINFORCE (no baseline)
python train.py --algo reinforce --episodes 800 --seed 42

# REINFORCE with baseline
python train.py --algo reinforce_baseline --episodes 600 --seed 42

# Actor-Critic
python train.py --algo actor_critic --episodes 600 --seed 42

# Custom learning rate
python train.py --algo actor_critic --episodes 500 --lr 0.003 --seed 123
```

### Options

- `--algo`: Algorithm choice (reinforce, reinforce_baseline, actor_critic)
- `--episodes`: Number of episodes (default: 1000)
- `--seed`: Random seed (default: 42)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--hidden_layers`: Network size (default: 128 128)

---

## Experiments

### Run Comparison

```bash
python experiments.py --num_seeds 5 --episodes 800
```

This runs all 3 algorithms with 5 different seeds and generates:
- `results/comparison_all_algorithms.png` - Learning curves
- `results/sample_efficiency.png` - Sample efficiency
- `results/comparison_statistics.csv` - Stats summary
- Individual CSVs for each run

### Reproducing Experiments

```bash
# Multi-seed ablation (REINFORCE, baseline, Actor-Critic)
python experiments.py --num_seeds 5 --episodes 800

# Gradient variance study (saves to results/gradient_analysis/)
python -c "from experiments import run_gradient_variance_analysis; run_gradient_variance_analysis(episodes=500, num_seeds=3)"
```

---

## Expected Results

CartPole-v1 learning progression:
- **0-200 episodes:** Early learning (return ~10-30)
- **200-500 episodes:** Active learning (return ~30-195)
- **500-800 episodes:** Solving (return ~195-500)

**Solved:** Average return ≥ 195 over last 100 episodes

| Algorithm | Episodes to Solve | Final Return |
|-----------|------------------|--------------|
| REINFORCE | 300-500 | 450-480 |
| REINFORCE+Baseline | 250-400 | 460-490 |
| Actor-Critic | 200-350 | 470-500 |

---

## How It Works

### Policy Gradient

All methods use:
```
∇J(θ) = E[∇log π(a|s) × G_t]
```

Where:
- `π(a|s)` = policy (action probabilities)
- `G_t` = return (or advantage)

### Key Differences

**REINFORCE:**
- Uses full Monte Carlo returns: `G_t = Σ γ^k * r_{t+k}`
- High variance

**REINFORCE+Baseline:**
- Uses advantage: `A_t = G_t - V(s_t)`
- Lower variance

**Actor-Critic:**
- Uses TD target: `r_t + γ*V(s_{t+1})`
- Lowest variance, some bias

---

## Hyperparameters

Default values:

```python
gamma = 0.99              # Discount factor
policy_lr = 0.001         # Policy learning rate
value_lr = 0.0005         # Value learning rate
hidden_layers = [128, 128] # Network size
entropy_coef = 0.01       # Entropy bonus (Actor-Critic)
```

---

## Google Colab

1. Upload Python files to Colab
2. Install: `!pip install gymnasium[classic-control] torch matplotlib numpy`
3. Run training commands
4. Or use `run_project.ipynb` (Open in Colab badge above)

---

## Analysis

For a theoretical comparison of variance, baselines, and bootstrapping (with pointers to the new plots and statistics), see [analysis.md](analysis.md).

---

## Troubleshooting

**Not learning?**
- Train longer (800+ episodes)
- Try different seed
- Adjust learning rate

**Slow training?**
- Normal for RL!
- Use GPU (auto-detected)
- Or use Colab

**High variance?**
- Run multiple seeds
- Use experiments.py
- Average results

---

