# Exploring Clustered Optimal Policies via Off-Policy Reinforcement Learning for Business Use Cases
A Practical Walk-Through from Single-Head Baselines to Adaptive Policy Clustering
In real-world marketing, customer responses to promotions vary - some favor deep discounts, while others prefer loyalty rewards or limited-time offers. This diversity makes it difficult for a single policy to maximize profit across the board. To tackle this, we evaluate four reinforcement learning (RL) pipelines, each trained on the same offline dataset. Their performance is estimated using Inverse Propensity Sampling (IPS), which re-weights each logged interaction by 1/plog​. This allows us to compare new policies fairly without re-running the original campaign.
Single-head DQN is our value-based baseline: one Q-network is trained on all records, producing a single universal policy.
Fixed-K DQN starts with K parallel heads; each head tries to specialize on a slice of the data and, at inference time, the system executes the action from the head whose Q-value is highest.
Adaptive-Clustered PPO takes a policy-gradient route. A lightweight gate assigns soft mixture weights π_gate(k∣s) to multiple actor–critic heads; gradients flow through these weights so each head gradually "owns" the customers it predicts.
Single-head PPO offers a unified policy-gradient counterpart to the DQN baseline, learning one stochastic policy from all samples.

With importance-sampling evaluation in place, we can fairly compare how soft vs. hard specialization, value- vs. policy-based updates, and static vs. adaptive routing influence expected return. The resulting insights help marketers deploy RL systems that personalize promotions at scale while remaining confident.
