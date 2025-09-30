DiZO Large Model Examples
=========================

This directory contains scripts and helpers to run DiZO (Discrete Zero-Order optimization) across a variety of modern LLM architectures.

Supported / Tested Architectures
--------------------------------
- OPT (facebook/opt-*)
- GPT-2 / GPT-Neo style (gpt2, Eleuther models)
- Llama / Llama2 / Llama3 (meta-llama/Llama-3.* family)
- Qwen / Qwen2 / Qwen2.5
- Mistral / Mixtral
- Falcon (tiiuae/falcon-*)
- (Experimental) BERT / RoBERTa style for classification (head tuning / linear probing)

Core Scripts
------------
- `dizo.sh` : Main launcher that wraps `run.py` with environment variables.
- `run.py`  : Unified training / evaluation / DiZO fine-tuning loop.
- `llama3_dizo_example.py` : End-to-end example specifically highlighting Llama3 integration.

Quick Start (SST2 Few-Shot DiZO Fine-tuning)
--------------------------------------------
Each command follows the pattern:  MODEL=... TASK=SST2 LR=... EPS=... STEPS=... bash dizo.sh

OPT (baseline from original paper):
```bash
MODEL=facebook/opt-1.3b TASK=SST2 LR=1e-6 EPS=1e-3 STEPS=1000 bash dizo.sh
```

Llama3 (1B / 3B / 8B example IDs change as needed):
```bash
MODEL=meta-llama/Llama-3.2-1B TASK=SST2 LR=5e-6 EPS=1e-3 STEPS=1000 bash dizo.sh
```

Falcon (consider smaller LR for stability on larger checkpoints):
```bash
MODEL=tiiuae/falcon-7b TASK=SST2 LR=2e-6 EPS=1e-3 STEPS=1000 bash dizo.sh
```

Qwen2.5 0.5B (already validated):
```bash
MODEL=Qwen/Qwen2.5-0.5B TASK=SST2 LR=3e-6 EPS=5e-4 STEPS=1000 bash dizo.sh
```

Mistral:
```bash
MODEL=mistralai/Mistral-7B-v0.1 TASK=SST2 LR=4e-6 EPS=1e-3 STEPS=1000 bash dizo.sh
```

Mixtral (larger mixture model – may need fewer batch tokens / more memory):
```bash
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 TASK=SST2 LR=4e-6 EPS=1e-3 STEPS=800 bash dizo.sh
```

Head / Linear Probing Only (freeze body, tune classification head):
```bash
MODEL=facebook/opt-1.3b TASK=SST2 LR=1e-5 EPS=1e-3 STEPS=500 MODE=ft bash dizo.sh --head_tuning
```

Key Environment Variables
-------------------------
- MODEL : HuggingFace model id
- TASK  : Task name (SST2, RTE, CB, BoolQ, etc.)
- LR    : Learning rate for DiZO update (learning_rate arg)
- EPS   : Perturbation epsilon for zero-order estimation (zo_eps)
- STEPS : Training max steps (max_steps)
- BS    : Per-device train batch size (per_device_train_batch_size)
- ENHANCED : Set to `zo` or `fo` to enable enhanced DiZO gamma search pathway

Recommended Hyperparameter Hints
---------------------------------
- Llama/Llama2/Llama3: LR 5e-6, EPS 1e-3, reduce batch size if OOM
- Qwen/Qwen2.x: LR 3e-6, EPS 5e-4 (slightly smaller for stability)
- Falcon: LR 2e-6–3e-6, EPS 1e-3
- Mistral: LR 4e-6, EPS 1e-3
- OPT (reference): LR 1e-6, EPS 1e-3

Troubleshooting
---------------
1. OOM Errors: Lower BS (BS=8 or 4) or reduce STEPS for quick smoke tests.
2. Tokenizer pad_token errors: Automatically handled; if still failing, manually export `--pad_token_id` via code modification.
3. Falcon weight tying / missing lm_head: Ensure using latest transformers; older versions may mismatch head names.
4. Mixtral MoE memory spikes: Try `--load_bfloat16` or int8 loading plus smaller BS.

Planned / Optional Extensions
------------------------------
- Automatic per-model hyperparameter injection (see `enhanced_model_support.get_dizo_hyperparameters`).
- FSDP + DiZO hybrid to scale to 70B+.
- Classification backbone (BERT/RoBERTa) scripts for GLUE full sweep.

License & Attribution
---------------------
Refer to the root project README for license details. Model weights remain subject to their original licenses (e.g., Llama3, Falcon, Qwen families).

Happy zero-order tuning! 🎯

