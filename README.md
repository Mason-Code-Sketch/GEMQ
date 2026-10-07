<div align="center">

<h1>GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2605.23078-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2605.23078)

</div>

GEMQ is a post-training quantization framework for Mixture-of-Experts (MoE) LLMs that enables extreme low-bit quantization (down to 1.5 bits per expert) with minimal accuracy degradation. It works by:
1. automatically assigning different bit-widths to experts based on their importance;
2. fine-tuning the routers so they can better work with quantized experts;
3. optionally using progressive quantization to refine the bit allocation.


### What's in this repo
* An ILP solver for global expert-level bit allocation
* GPTQ-based quantization and router fine-tuning pipelines
* Efficient low-bit MoE triton kernels for **real** quantized inference


## Updates

- [2026/08] Bit allocation now runs on **HiGHS**, the ILP solver bundled with SciPy, so regenerating the bit configs no longer needs a Gurobi license. Gurobi stays available as an optional backend.
- [2026/08] Real quantized inference now covers **OLMoE-1B-7B-0924** and **Qwen3-30B-A3B**, alongside Mixtral-8x7B and DeepSeek-V2-Lite. Run it with `scripts/bench_generate_<model>.sh`.
- [2026/08] Real quantization is verified to match fake quantization end to end -- a 0.06% perplexity gap on DeepSeek-V2-Lite and 0.03% on OLMoE-1B-7B-0924. Run the checks with `scripts/test_real_quant.sh`.
- [2026/08] Fixed a ~15% perplexity regression on DeepSeek-V2 caused by a missing YaRN `mscale` in HF's built-in implementation ([transformers#47435](https://github.com/huggingface/transformers/pull/47435)).


## Installation

```bash
conda create -n gemq python=3.10 -y
conda activate gemq
git clone https://github.com/jndeng/GEMQ
cd GEMQ
pip install -e .

# (Optional) only needed if you want to solve the bit allocation with Gurobi
# instead of the default HiGHS solver (and thus requires a Gurobi license):
# pip install -e ".[gurobi]"
```

> [!NOTE]
>
> By default, bit allocation is solved with **HiGHS**, which does not require a commercial license.
> In our experiments, however, we used **Gurobi** to produce the configs under `configs/`.
> Gurobi remains available as an optional backend -- install it as shown above, then set `ilp_backend="gurobi"` in `scripts/allocate_<model>.sh`.


## Usage

> `scripts` provides the full pipeline -- bit allocation, quantization and real quantized inference -- for **Mixtral-8×7B**, **DeepSeek-V2-Lite**, **OLMoE-1B-7B-0924** and **Qwen3-30B-A3B**.


### 1. Bit Allocation

> [!NOTE]
>
> We provide pre-generated bit allocation configs under `configs`, which can be used directly for quantization. You may skip this section if you do not want to regenerate them.

> [!IMPORTANT]
>
> **All provided configs and results reported in the paper were produced with the Gurobi backend.** HiGHS was added later solely to remove the Gurobi license requirement. Both backends solve the same ILP, but because the optimum may not be unique, HiGHS can return a different allocation. To reproduce the paper exactly, use the provided configs or set `ilp_backend="gurobi"` in the allocation script.

To generate the configs from scratch, follow the steps below.


1. Download the first shard of the C4 training dataset (c4-train.00000-of-01024.json) from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz) and save it under `./data`.

2. Run `scripts/compute_stats_<model>.sh` to compute model statistics on the calibration dataset. The resulting statistics (gradients and perturbation errors) will be saved under `cache`.


3. Run `scripts/allocate_<model>.sh` to solve the ILP for bit allocation using the generated model statistics. The allocation results (bit configs) will be saved under `configs`. 


### 2. Mixed-Precision Quantization

Simply run `scripts/quantize_<model>.sh` for model quantization. Please refer to the script for the detailed available options.

The evaluation code runs automatically after quantization. If you want to evaluate the model on downstream tasks, please ensure that [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is installed.

Quantized models will be saved under `results`.

### Evaluate an Existing Checkpoint

Install the optional evaluation dependencies in the model's existing environment:

```bash
pip install -e ".[eval]"
```

Evaluate a saved real-quant checkpoint, including its fine-tuned Router, without
running quantization or Router training again:

```bash
python -m gemq.eval_zeroshot --model_path "$CHECKPOINT_DIR" --batch_size auto --max_batch_size 8
```

The default suite is PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, MathQA,
and MMLU. It uses LM Evaluation Harness 0.4.13, zero shots, the native task prompts,
and no chat template. Scores use `acc_norm` when available, otherwise `acc`.
MMLU counts once in the equally weighted seven-task average. `--include_gsm8k`
adds separate strict/flexible exact-match scores without changing that average.

Results and logs default to `zeroshot.json` and `zeroshot.log` inside the model
directory, beside `qmodel.pt`. `--output` selects a different new JSON path;
neither existing results nor logs are overwritten. The JSON includes task scores,
the full harness result, settings, and package versions. Use `--is_fp` for a
floating-point HF model directory. Keep the checkpoint's model implementation
and Transformers version when selecting the evaluation environment.

For a terminal-only check, use `--limit` and omit `--output`:

```bash
python -m gemq.eval_zeroshot --model_path "$CHECKPOINT_DIR" \
  --tasks piqa --limit 4 --batch_size 1
```

This evaluates four examples per selected leaf task and creates no result JSON
or log. For MMLU, the limit applies separately to each subject. Formal evaluation
omits `--limit`; a partial task suite does not produce a seven-task average.


### 3. Inference

Use `scripts/bench_generate_<model>.sh` to run inference demos and benchmark the real quantized models. Set `bpe` and `finetune_routers` there to match the quantization run, since the checkpoint path is derived from them.

> [!NOTE]
>
> Decoding is fully fused; prefill still loops over hit experts in Python, so its throughput is dominated by kernel launch overhead and scales with depth and expert count rather than with prompt length.


## License

Released under the [MIT License](LICENSE).

## Acknowledgements
This repository builds upon several excellent open-source projects, including [MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE), [GPTQ](https://github.com/IST-DASLab/gptq), [HQQ](https://github.com/dropbox/hqq), [GemLite](https://github.com/dropbox/gemlite), and [gpt-fast](https://github.com/meta-pytorch/gpt-fast). We sincerely thank the authors and contributors for making their code publicly available.

## Citation
If you find GEMQ useful for your research or project, please consider citing:
```bibtex
@article{deng2026gemq,
  title={GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs},
  author={Deng, Jianing and Wang, Song and Wang, Dongwei and Liu, Zijie and Chen, Tianlong and Yang, Huanrui and Hu, Jingtong},
  journal={arXiv preprint arXiv:2605.23078},
  year={2026}
}
```
