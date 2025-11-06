# README for prompt tuning mamba

## Files used for customized initial state

1. Model running script: [benchmarks/benchmark_generation_mamba_simple_mamba2_init_state.py](/root/autodl-tmp/mamba/benchmarks/benchmark_generation_mamba_simple_mamba2_init_state.py)


```python


#line58-line81
mixer_kwargs = {}
if is_mamba and args.use_custom_init_state:
    print("Generating a custom random initial state...")
    # 从模型中找到一个 Mamba2 层来获取状态维度
    first_mamba2_layer = None
    for layer in model.backbone.layers:
        if hasattr(layer, "mixer") and "Mamba2" in layer.mixer.__class__.__name__:
            first_mamba2_layer = layer.mixer
            break
    
    if first_mamba2_layer:
        nheads = first_mamba2_layer.nheads
        headdim = first_mamba2_layer.headdim
        d_state = first_mamba2_layer.d_state
        
        # random initial state
        #在这里指定initial state
        custom_initial_state = torch.randn(
            args.batch, nheads, headdim, d_state, device=device, dtype=dtype
        )
        
        mixer_kwargs["initial_states"] = custom_initial_state
        print(f"Custom initial state created with shape: {custom_initial_state.shape}")
    else:
        print("Warning: --use-custom-init-state was set, but no Mamba2 layer was found. Ignoring.")


```

2. Mixer model definition：[mamba_ssm/models/mixer_seq_simple_init_state.py](mamba/mamba_ssm/models/mixer_seq_simple_init_state.py)

3. Mamba2 block definition: [mamba_ssm/modules/mamba2_init_states.py](mamba/mamba_ssm/modules/mamba2_init_states.py)

4. Generation method definition: [mamba_ssm/utils/generation_init_state.py](mamba/mamba_ssm/utils/generation_init_state.py)

## To run the script 

use "--use-custom-init-state" flag to decide whether a custom initial state is used.

```sh
python benchmarks/benchmark_generation_mamba_simple_mamba2_init_state.py --model-name "state-spaces/mamba2-2.7b" --prompt "Hello" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --use-custom-init-state
```