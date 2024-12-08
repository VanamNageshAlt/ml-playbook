# LoRA tuning (cheat)

## Rank (`r`)

- 8 -- minimum for most useful adaptation
- 16 -- common default, good first try
- 32-64 -- larger tasks or when you need more capacity
- 128+ -- diminishing returns, mostly noise

## Alpha

Default `alpha = 2 * r`. Effective scale on adapter outputs is `alpha / r`.
Setting `alpha = r` halves it; `alpha = 4*r` doubles it.

## target_modules

For LLaMA-style: `q_proj, k_proj, v_proj, o_proj` is the standard.

Adding MLP projections (`gate_proj, up_proj, down_proj`) can help on harder
tasks but doubles trainable param count.

## Dropout

`lora_dropout` 0.0-0.1 typical. 0.05 default. Higher dropout helps prevent
overfitting on small datasets.

## QLoRA combo

4-bit base model + LoRA = QLoRA. Same recipe, just load base with
`bitsandbytes` 4-bit quantization first.
