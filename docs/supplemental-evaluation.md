# Supplemental MMLU and GSM8K evaluation

The two independent profiles are `mmlu_5shot` and `gsm8k_5shot`. The existing seven-task command and outputs keep their zero-shot protocol. Supplemental scores never enter its seven-task average.

Protocol: lm-eval 0.4.13; seed 0 by default; no chat template or added BOS. MMLU uses the harness dev-set first-five examples and sample-weighted accuracy over 57 subjects / 14,042 test questions. GSM8K uses five training demonstrations, the harness Question/Answer prompt and answer filters, deterministic generation, and an explicit 512-token generation budget. Both strict-match and flexible-extract exact match are stored for all 1,319 test questions. The resolved task configs, including stop strings and extraction filters, are retained in `harness.configs` with a digest in `protocol`.

Each task is a separate invocation and result. Batch size defaults to `auto`, with model-specific caps listed below. Use `--batch_size 1` (or another positive integer) for a fixed batch, and `--max_batch_size 4` to lower the automatic cap. The cap applies to automatic sizing; an explicit fixed batch is used as requested. Actual batch sizes depend on available memory and input lengths. A smoke `--limit 1` means one question per leaf task (57 questions for MMLU), never a full benchmark result. It does not write a formal result JSON. Full runs check actual shot counts, effective sample counts and metrics before writing. Existing results/logs are refused. This interface does not schedule experiments.

Use the existing entry point:

```bash
python -m gemq.eval_zeroshot --model_path /absolute/checkpoint/path --evaluation mmlu_5shot --log-path /absolute/batch/logs/model-mmlu-5shot-bit.log
python -m gemq.eval_zeroshot --model_path /absolute/checkpoint/path --evaluation gsm8k_5shot --log-path /absolute/batch/logs/model-gsm8k-5shot-bit.log
```

Results default to `mmlu_5shot.json` and `gsm8k_5shot.json` beside `ppl.json` and `zeroshot.json`. Resource JSONs are stored beside each new result. Existing backend, dtype and layer-device flags remain available. `--is_fp` selects a floating HF model. Packed-file size/mtime and small config hashes are recorded; this is not an additional full weight hash.

The existing seven-task zeroshot entry also defaults to `auto` with the model-specific cap listed below; its batch options accept the same manual overrides.

For GSM8K auto batching, the probe length is the longest effective request input plus its generation budget (512 tokens), capped by the unchanged model context limit. The native five-forward-pass probe and OOM batch reduction are retained; this is not a full generation benchmark. Formal input truncation, generation and scoring are unchanged. `GSM8K_BATCH_PROBE` logs and `protocol.generation_batch_probes` record the lengths and selected batch. Fixed batch settings bypass this probe.

### Model-specific auto batch defaults

The loaded model config selects the auto cap. An explicit maximum batch argument overrides these defaults; a fixed batch remains explicit.

| Loaded `model_type` | Seven tasks | MMLU 5-shot | GSM8K 5-shot |
|---|---:|---:|---:|
| `deepseek_v2` | 32 | 24 | 128 |
| `qwen2_moe` | 32 | 24 | 128 |
| `mixtral` | 34 | 13 | 48 |
| `qwen3_moe` | 51 | 60 | 96 |
| Other models | 51 | 17 | 96 |

DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B defaults are initial estimates for four 24 GB RTX 4090 cards with decoder layers resident across the cards. Selection uses `model_type`, without hardware detection. Historical GEMQ `grouped_cached` seven-task runs completed at batch 8 for both models; Qwen1.5 batch 12 and both models' MMLU/GSM caps remain estimates pending GPU validation. Different backends or placement may need an explicit cap. Mixtral 34 passed complete KDMoQ seven-task evaluation; its 13 and 48 caps remain estimates. Tasks, few-shot settings and generation budgets are unchanged.


### MMLU scored positions

MMLU uses compact scoring for native Transformers Qwen2MoE, Qwen3MoE, Mixtral and DeepSeek-V2 models with a standard output head, on a single evaluation process (including layer placement across devices). For single-token answers a temporary output-head hook selects each batch row's valid final positions on the hidden tensor's actual device, then normalizes each scored row over the full vocabulary. CPU, single-device, sharded and offloaded placement use the same path. Native context truncation, right padding, option reuse, greedy flags and request ordering are retained. Multi-token answers and unsupported models use native HFLM scoring. Auto probes use the same compact scoring path, five forwards and native OOM batch reduction. They cover up to batch-size distinct scoring positions; temporary probe outputs are released each iteration. `protocol.mmlu_scored_positions` records activation, optimized request count and fallback calls.

Qwen1.5 MMLU temporarily disables KV caching and restores the original setting on success or exception. GSM8K retains its existing generation cache behavior. CPU tests with small real MoE models check scores and selected answers against native HFLM; full-model GPU speed and peak memory require separate measurements before claiming a performance gain.

DeepSeek checkpoint-provided implementations use a temporary output-head input hook during compact scoring, retaining their original decoder and FP32 logits cast. The hook is removed even on errors. This path is CPU correctness-tested; full-model DeepSeek GPU memory savings remain unmeasured.

MMLU caps: DeepSeek-V2 24 and Qwen1.5-MoE 24 for four RTX 4090 cards; Qwen3-MoE 60 and Mixtral 13 for DGX Spark. These are estimated auto upper limits, not GPU-validated capacities. Other profiles and explicit overrides are unchanged.
