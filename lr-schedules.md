# LR schedules (cheat)

## Defaults

- **Cosine** -- standard for fixed-step training. Smooth ramp from peak to ~0.
- **Linear with warmup** -- transformer LM training default (warm 5-10% of steps).
- **Constant + warmup** -- fine-tuning with low compute budget.
- **Reduce on plateau** -- ergonomic for vision transfer learning.

## Picking warmup steps

Rule of thumb: 5-10% of total steps. Smaller for short fine-tuning runs (50-200 steps).

## Decay endpoint

End at 1-10% of peak LR for cosine. Don't decay all the way to 0 -- the
last few steps end up doing nothing.

## Common mistake

Resuming training without restoring the scheduler state. The LR resets to
initial peak, often much higher than where it should be.
