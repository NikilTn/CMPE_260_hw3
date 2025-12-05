# Policy Gradient Variance & Baseline Notes (CartPole-v1)

I ran 5 seeds each for REINFORCE, REINFORCE+Baseline, and Actor-Critic and then checked the gradient variance plot (`results/gradient_analysis/gradient_variance.png`). REINFORCE shows the noisiest curve. That makes sense because its Monte Carlo return ties every update to the whole episode, so any early bad moves echo through the gradient. It feels jumpy across seeds.

When I plug in a value baseline, the curve drops. Subtracting \(V(s_t)\) basically recenters the target, so the policy gradient stays unbiased but wiggles less. You can see the learning curves in `results/comparison_all_algorithms.png` look tighter, and the confidence bands aren’t as wide.

Actor-Critic goes further by bootstrapping with \(r_t + \gamma V(s_{t+1})\). Now the update uses a short lookahead plus the critic, so variance shrinks again, but we accept some bias from the value estimate. In practice on CartPole, that trade-off seems worth it: the variance line for Actor-Critic stays below REINFORCE in the plot, and in `results/comparison_statistics.csv` it reaches the solve threshold faster and finishes with solid average returns. Overall takeaway: REINFORCE is high variance, baseline fixes a good chunk without bias, and bootstrapping lowers variance even more with small bias that pays off here.

