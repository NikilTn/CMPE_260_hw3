[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NikilTn/CMPE_260_hw3/blob/master/run_project.ipynb)

# Policy Gradient Methods - CartPole-v1

Three simple policy gradient tricks I coded: REINFORCE, REINFORCE+Baseline, and Actor-Critic.

---

## Quick Start (what I run)

```bash
# Install deps
pip install -r requirements.txt

# (Optional) 
# python test_quick.py

# Train one run
python train.py --algo reinforce --episodes 500 --seed 42
python train.py --algo actor_critic --episodes 500 --seed 42

# Compare all algorithms
python experiments.py --num_seeds 5 --episodes 800
```

---


## Algorithms 

1) REINFORCE: plain Monte Carlo policy gradient, super simple, high variance.  
2) REINFORCE + baseline: same thing but subtract V(s) so it shakes less.  
3) Actor-Critic: TD target with a critic, updates every step, lowest variance but a little bias.

---

## Training Commands

### Basic Usage

```bash
python train.py --algo ALGORITHM --episodes NUM --seed SEED
```

### Examples I tried

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

### Options (the ones I actually touch)

- `--algo`: Algorithm choice (reinforce, reinforce_baseline, actor_critic)
- `--episodes`: Number of episodes (default: 1000)
- `--seed`: Random seed (default: 42)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--hidden_layers`: Network size (default: 128 128)

---

## Experiments

### Run Comparison (multi-seed)

```bash
python experiments.py --num_seeds 5 --episodes 800
```

This runs all 3 algos with 5 seeds and drops:
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

## Expected Results (rough)

- 0-200 eps: learning starts (returns ~10-30)  
- 200-500 eps: ramp up (returns ~30-195)  
- 500-800 eps: usually solves (195-500)

Solved = avg return ≥ 195 over last 100 eps.

---

## How It Works (short take)

All use ∇log π(a|s) * return/advantage.  
- REINFORCE: full Monte Carlo return → high variance, no bias.  
- REINFORCE+Baseline: use advantage G - V(s) → same mean, less variance.  
- Actor-Critic: TD target r + γV(s') → lowest variance, a bit of bias from the critic.

---

## Hyperparameters (defaults I used)

```python
gamma = 0.99
policy_lr = 0.001
value_lr = 0.0005
hidden_layers = [128, 128]
entropy_coef = 0.01
```

---

## Google Colab

1) Upload the files or clone.  
2) Install: `!pip install gymnasium[classic-control] torch matplotlib numpy`  
3) Run the commands above or just open `run_project.ipynb` (badge at top).

---

## Analysis

I wrote up the variance/baseline/bootstrapping notes in [analysis.md](analysis.md) with links to the plots.

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

