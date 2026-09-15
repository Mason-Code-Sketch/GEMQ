# Local Protocol Configs

Each file defines one model using paths relative to the GEMQ repository root.
`MODEL_PATH` resolves under `../../../data/models` and `DATASET_ROOT` resolves
under `../../../data/datasets` when `scripts/run_protocol.sh` runs.

The runner stores statistics, allocations, and fake-quantized checkpoints under
`results/protocol/<config-name>/`.

```bash
scripts/run_protocol.sh configs/protocol/deepseek-v2-lite.env bootstrap 3
scripts/run_protocol.sh configs/protocol/deepseek-v2-lite.env progressive 3 2.5
scripts/run_protocol.sh configs/protocol/deepseek-v2-lite.env progressive 2.5 2
```

Set `FINETUNE_ROUTERS=true` in a model config to enable GEMQ router fine-tuning
for the quantization command. The associated `RFT_*` fields retain GEMQ's
original defaults unless explicitly overridden.

`bootstrap` computes the base-model statistics, allocation, fake GPTQ model,
and Router FT checkpoint. `progressive` computes statistics from the preceding
fake-quantized and Router-FT checkpoint, then always quantizes from the
original FP model configured by `MODEL_PATH`. Every stage writes a manifest
under `results/protocol/<config-name>/manifests`.
