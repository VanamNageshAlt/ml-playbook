# Eval honesty (cheat)

## Common ways your eval lies

1. **Train/test contamination**: your "eval set" leaked into pretraining data.
2. **Cherry-picked checkpoints**: reporting best across many runs.
3. **Cherry-picked seeds**: only showing the seed that worked.
4. **Tiny eval sets**: 200 examples != stable estimate.
5. **Single-metric tunnel vision**: top-1 alone hides calibration / fairness issues.

## Things to do

- **Multiple seeds**: report mean and std (or IQM) over 3+ seeds.
- **Confidence intervals**: bootstrap CIs are cheap and informative.
- **Stratified eval**: break down by class / domain / difficulty bucket.
- **Hold out a *separate* test set**: never tune anything on it. Touch it only once at the end.
- **Pre-register your protocol**: write down the eval before you run it.

## The single most useful question

> If I ran this with a different seed tomorrow, what would I expect the result to look like?
