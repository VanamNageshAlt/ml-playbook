# GPU debugging (cheat)

## OOM

1. Reduce `batch_size`. (Obvious but worth checking first.)
2. Enable gradient accumulation: `effective_batch = batch * accum_steps`.
3. Enable gradient checkpointing on the model.
4. Switch to bf16 / fp16 if not already.
5. Use 8-bit optimizers (`bitsandbytes.AdamW8bit`).
6. Profile: `torch.cuda.memory_summary()` shows allocator state.

## Slow training

1. Check `nvidia-smi` -- is GPU actually saturated? If <60%, it's CPU/IO-bound.
2. Increase `num_workers` in DataLoader.
3. `pin_memory=True` and `non_blocking=True`.
4. Prefetch with WebDataset or similar for streaming pipelines.
5. Profile with `torch.profiler` to find the actual bottleneck.

## NaN losses

1. Lower learning rate first.
2. Add gradient clipping if not already (norm 1.0 to start).
3. Check for divisions, log(0), sqrt(negative) in your loss.
4. fp16 + mixed precision: missing GradScaler? bf16 instead?
5. Bad data: NaN in inputs, infinity in labels.

## Debugging command

```bash
CUDA_LAUNCH_BLOCKING=1 python -X faulthandler train.py
```
Forces sync GPU ops so the stack trace points at the actual line.
