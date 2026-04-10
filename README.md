<div align="center">
<h1>Hybrid Policy Distillation</h1>

[<img src="https://img.icons8.com/emoji/48/000000/open-book-emoji.png" width="18" style="vertical-align:middle; margin-right:6px"/> <u>Paper</u>](https://arxiv.org/abs/2508.17784) |
[<img src="https://img.icons8.com/material-outlined/24/000000/github.png" width="18" style="vertical-align:middle; margin-right:6px"/> <u>Code</u>](https://github.com/zwhong714/PSFT) |
[<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="18" style="vertical-align:middle; margin-right:6px"/> <u>Model</u>](https://huggingface.co/collections/wh-zhu/psft-68afb02eb237149f9bd9075e)
</div>

This repository contains the project materials and code for Hybrid Policy Distillation (HPD). If you find it useful in your research or work, please consider giving it a star and citing it.

We currently provide **two implementations**: LlamaFactory version and verl version. 



## Implementations

| Implementation | Location | Training Regime | Main Entry | Notes |
| --- | --- | --- | --- | --- |
| LlamaFactory | `LlamaFactory/` | HPD loss | `llamafactory-cli train ...` | Supports full fine-tuning and LoRA |
| verl | `verl/recipe/HPD/` | RL / post-training | `bash recipe/HPD/run_hpd.sh` | Current recipe targets `fsdp` / `fsdp2` |


## Installation

### 1. LlamaFactory Version

Run the following inside `LlamaFactory/`:

```bash
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

If you plan to run full-parameter HPD with DeepSpeed:

```bash
pip install -r requirements/deepspeed.txt
```

For more environment details, see `LlamaFactory/README.md`.

### 2. verl Version

Run the following inside `verl/`:

```bash
cd verl
pip install -e .
```

Depending on your backend, you may also need optional dependencies such as `vllm`, `flash-attn`, or math-evaluation packages. For backend-specific environment setup, see `verl/README.md`.

## LlamaFactory Version

The LlamaFactory implementation integrates HPD directly into the SFT workflow.

### Key Files

- `LlamaFactory/src/llamafactory/train/hpd.py`: HPD loss implementation.
- `LlamaFactory/src/llamafactory/train/sft/trainer.py`: trainer-side HPD integration.
- `LlamaFactory/src/llamafactory/hparams/finetuning_args.py`: HPD-related arguments such as `use_hpd_loss`.
- `LlamaFactory/examples/train_full/qwen3_full_hpd.yaml`: full-parameter HPD example config.
- `LlamaFactory/tests/train/test_sft_trainer.py`: regression test covering HPD loss logging.

### Quick Start: Full-Parameter HPD

```bash
cd LlamaFactory
llamafactory-cli train examples/train_full/qwen3_full_hpd.yaml
```

This example uses:

- `stage: sft`
- `finetuning_type: full`
- `use_hpd_loss: true`

For full-parameter fine-tuning, if `ref_model` is not set, HPD uses a frozen copy of `model_name_or_path` as the teacher model.

If you want to use an external teacher, set:

```yaml
ref_model: path_or_hf_repo_of_teacher
ref_model_adapters: path_to_teacher_adapter
ref_model_quantization_bit: 4
```

### Quick Start: LoRA HPD

LoRA is also supported, but unlike full fine-tuning, LoRA HPD requires an explicit teacher:

```bash
cd LlamaFactory
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
    use_hpd_loss=true \
    ref_model=Qwen/Qwen3-4B-Instruct-2507 \
    output_dir=saves/qwen3-4b/lora/hpd
```

### Practical Notes

- `use_hpd_loss` is only valid for `stage: sft`.
- LoRA HPD requires `ref_model`; full-parameter HPD does not.
- Full-parameter HPD keeps both student and teacher in memory, so GPU memory usage is much higher than standard SFT.
- The provided full example uses DeepSpeed ZeRO-3 by default for that reason.
- If you only want to test the pipeline, reduce `max_samples`, `cutoff_len`, and batch size first.

## verl Version

The `verl` implementation keeps HPD as a **recipe-local extension** instead of patching core `verl` PPO workers.

### Key Files

- `verl/recipe/HPD/main_hpd.py`: recipe entry point.
- `verl/recipe/HPD/hpd_trainer.py`: HPD trainer built on top of `RayPPOTrainer`.
- `verl/recipe/HPD/dp_actor.py`: HPD actor logic.
- `verl/recipe/HPD/fsdp_workers.py`: FSDP worker integration.
- `verl/recipe/HPD/config/hpd_trainer.yaml`: base Hydra config.
- `verl/recipe/HPD/run_hpd.sh`: runnable experiment script.

### Quick Start

```bash
cd verl
bash recipe/HPD/run_hpd.sh
```



### Current Constraints

- Current implementation targets `fsdp` / `fsdp2`.
- HPD currently expects `actor_rollout_ref.model.use_remove_padding=True`.
- The provided `run_hpd.sh` script uses `vllm` rollout and local absolute paths for the student model, reference model, training parquet, and AIME validation parquet files.

Before running, update at least the following variables in `verl/recipe/HPD/run_hpd.sh`:

- `MODEL_PATH`
- `REF_MODEL_PATH`
- `TRAIN_FILE`
- `test_path_1`
- `test_path_2`
- `CKPTS_DIR`

## Tasks And Data

### Math Reasoning

The reasoning setup is based on the math reasoning dataset hosted at:

- `https://huggingface.co/datasets/wh-zhu/train_openr1_8k`

The provided `verl` recipe script evaluates on AIME-style parquet files and is currently configured with local validation paths for `aime-24` and `aime-2025`.

### Coding

The LlamaFactory HPD example is currently wired for coding SFT:

- dataset name: `qwen_code`
- dataset registration: `LlamaFactory/data/dataset_info.json`
- current data file: `LlamaFactory/data/train_code_teacher_sft.parquet`

This repository also includes `LlamaFactory/data/deepseek_code_sft.parquet` as an additional coding dataset artifact.


## Citation

If you use this repository or the HPD idea in your work, please cite the paper linked above.
