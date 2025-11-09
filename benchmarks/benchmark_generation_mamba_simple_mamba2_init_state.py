# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple_init_state import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--use-custom-init-state", action="store_true", help="If set, generate a random initial state and pass it to the model.")
parser.add_argument("--debug-init-state", action="store_true", help="If set, record and print which layers used custom vs zero initial state.")
parser.add_argument("--init-state-layer", type=int, default=None, help="指定应用自定义初态的 Mamba2 层的 layer_idx；默认选取首个 Mamba2 层")
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen



mixer_kwargs = {}
if is_mamba and args.use_custom_init_state:
    print("Generating a custom random initial state...")
    # 根据用户指定 layer_idx 或默认选取首个 Mamba2 层
    target_mamba2_layer = None
    for layer in model.backbone.layers:
        if hasattr(layer, "mixer") and "Mamba2" in layer.mixer.__class__.__name__:
            if args.init_state_layer is None:
                target_mamba2_layer = layer.mixer
                break
            else:
                if getattr(layer, "layer_idx", None) == args.init_state_layer:
                    target_mamba2_layer = layer.mixer
                    break
    if target_mamba2_layer is None and args.init_state_layer is not None:
        # 用户指定的 layer 未找到，回退到首个 Mamba2 层并提示
        print(f"[warn] 指定的 --init-state-layer={args.init_state_layer} 未找到，对应层可能不是 Mamba2 或不存在，改为使用首个 Mamba2 层。")
        for layer in model.backbone.layers:
            if hasattr(layer, "mixer") and "Mamba2" in layer.mixer.__class__.__name__:
                target_mamba2_layer = layer.mixer
                break

    if target_mamba2_layer:
        nheads = target_mamba2_layer.nheads
        headdim = target_mamba2_layer.headdim
        d_state = target_mamba2_layer.d_state

        custom_initial_state = torch.randn(
            args.batch, nheads, headdim, d_state, device=device, dtype=dtype
        )

        layer_idx = getattr(target_mamba2_layer, "layer_idx", None)
        if layer_idx is not None:
            mixer_kwargs["initial_states_by_layer"] = {layer_idx: custom_initial_state}
            print(f"Custom initial state created for layer {layer_idx} with shape: {custom_initial_state.shape}")
        else:
            mixer_kwargs["initial_states"] = custom_initial_state
            print(f"Custom initial state created (no layer_idx found) with shape: {custom_initial_state.shape}")

        # 若开启调试，要求各层在 prefill 时记录是否使用了自定义初态
        # 用于检测自定义初态传递是否正确实现
        if args.debug_init_state:
            mixer_kwargs["debug_mark_init"] = True
    else:
        print("Warning: --use-custom-init-state was set, but no Mamba2 layer was found. Ignoring.")




if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        # 当传入 debug-init-state 标志时，prefill 阶段不使用 cg，防止报错
        cg=(not args.debug_init_state),
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
        **mixer_kwargs,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))



# 调试打印：显示哪些层使用了自定义初态，哪些层为零初态
if is_mamba and args.debug_init_state:
    print("==== Debug init state report (scheme D) ====")
    any_found = False
    for layer in model.backbone.layers:
        if hasattr(layer, "mixer") and hasattr(layer.mixer, "_init_state_debug"):
            info = layer.mixer._init_state_debug
            print(f"layer {info['layer_idx']}: {info['kind']}, init_sum={info['init_sum']:.4f}, shape={info['shape']}")
            any_found = True
    if not any_found:
        print("[debug-init-state] 未找到任何 _init_state_debug 标记，可能未使用 Mamba2 或未触发 forward。")



torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
