# Supplemental MMLU and GSM8K evaluation

The two independent profiles are `mmlu_5shot` and `gsm8k_0shot`. The existing seven-task command and outputs keep their zero-shot protocol. Supplemental scores never enter its seven-task average.

Protocol: lm-eval 0.4.13; seed 0 by default; no chat template or added BOS. MMLU uses the harness dev-set first-five examples and sample-weighted accuracy over 57 subjects / 14,042 test questions. GSM8K uses zero demonstrations, the harness Question/Answer prompt and answer filters, deterministic generation, and an explicit 256-token generation budget (the HFLM default). Both strict-match and flexible-extract exact match are stored for all 1,319 test questions. The resolved task configs, including stop strings and extraction filters, are retained in `harness.configs` with a digest in `protocol`.

Each task is a separate invocation and result. Batch size defaults to `auto`, capped at 64. Use `--batch_size 1` (or another positive integer) for a fixed batch, and `--max_batch_size 4` to lower the automatic cap. The cap applies to automatic sizing; an explicit fixed batch is used as requested. Actual batch sizes depend on available memory and input lengths. A smoke `--limit 1` means one question per leaf task (57 questions for MMLU), never a full benchmark result. It does not write a formal result JSON. Full runs check actual shot counts, effective sample counts and metrics before writing. Existing results/logs are refused. This interface does not schedule experiments.

Use the existing entry point:

```bash
python -m gemq.eval_zeroshot --model_path /absolute/checkpoint/path --evaluation mmlu_5shot --log-path /absolute/batch/logs/model-mmlu-5shot-bit.log
python -m gemq.eval_zeroshot --model_path /absolute/checkpoint/path --evaluation gsm8k_0shot --log-path /absolute/batch/logs/model-gsm8k-0shot-bit.log
```

Results default to `mmlu_5shot.json` and `gsm8k_0shot.json` beside `ppl.json` and `zeroshot.json`. Resource JSONs are stored beside each new result. Existing backend, dtype and layer-device flags remain available. `--is_fp` selects a floating HF model. Packed-file size/mtime and small config hashes are recorded; this is not an additional full weight hash.

The existing seven-task zeroshot entry also defaults to `auto` with a maximum batch size of 64; its batch options accept the same manual overrides.
