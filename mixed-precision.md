# Mixed precision (cheat)

## bf16

- Wider exponent than fp16 -- fewer overflow problems.
- No GradScaler needed.
- Requires Ampere+ on NVIDIA, MI200+ on AMD, TPU v3+.

## fp16

- More mantissa than bf16 -- slightly higher numerical precision.
- Needs `torch.cuda.amp.GradScaler` to avoid underflow on small gradients.
- Works on older GPUs.

## When to use which

- **Default**: bf16 if your hardware supports it.
- **Fall back to fp16**: pre-Ampere hardware (T4, V100, P100).
- **Stay in fp32**: numerically sensitive ops -- often wrap loss computation.

## Common gotcha

`autocast` covers forward + backward. Optimizer step should happen in fp32.
With bf16, master weights stay fp32 by default; with fp16, GradScaler handles
the loss scaling automatically.
