# RL implementation tips

## DQN gotchas

- **Target network update**: hard copy every N steps OR Polyak averaging. Don't skip.
- **Replay buffer warmup**: don't start training until buffer has ~1000+ transitions.
- **Loss**: smooth L1 (Huber) > MSE. Outliers in TD error happen.
- **Gradient clipping**: norm 10 is fine; tighter (1-5) sometimes helps stability.
- **Epsilon decay**: linear from 1.0 -> 0.05 over ~10-20% of steps, then constant.

## PPO gotchas

- **Advantage normalization**: do it per-batch. Forgetting this kills performance.
- **Value function loss coefficient**: 0.5 is the standard; 1.0 sometimes helps.
- **Entropy bonus**: 0.01 standard for discrete; 0.0 often fine for continuous.
- **Clip coefficient**: 0.2 from the paper. Don't over-think this.
- **Number of epochs per rollout**: 4 is common; too many hurts more than helps.
- **Orthogonal init**: matters more than people think. sqrt(2) on hidden, 0.01 on policy head, 1.0 on value head.

## General

Watch the KL divergence between old and new policy. Sudden spikes mean LR
or clip is misconfigured.
