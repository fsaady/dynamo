<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.6-35B-A3B Recipe Performance

These results measure the aggregated recipes using [AIPerf](https://github.com/ai-dynamo/aiperf) 0.10.0 and the
[`64k_400_90kv_agent_new_noschedule_short_15perc.jsonl`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl)
trace.

## Workload

The measured workload uses 64K input tokens, 400 output tokens, 90% KV reuse,
concurrency 32, and 3,541 trace requests. The trace
contains 131 requests that exceed the model context length; those are expected
to fail. Successful-request metrics therefore contain 3,410 requests.

```bash
aiperf profile \
  -m nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --input-file /data/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl \
  --custom-dataset-type mooncake_trace \
  --num-requests 3541 \
  --url http://<deployment>-frontend:8000 \
  --endpoint-type chat \
  --streaming --use-server-token-count \
  --extra-inputs ignore_eos:true \
  --concurrency 32 --workers-max 32 \
  --random-seed 42 --ui none \
  --tokenizer-trust-remote-code \
  --request-timeout-seconds 1200
```

Run one separate 32-request pass using the same trace and options before the
measured phase for warmup.

## Results

All measurements use the Dynamo SGLang 1.4.0 runtime image and one GPU.

| Recipe | Model | GPU | Passed | Failed | Output tok/s/GPU | Average tok/s/user | P50 tok/s/user | P50 TTFT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`sglang/agg-b200-mtp/deploy.yaml`](../sglang/agg-b200-mtp/deploy.yaml) | NVFP4 | 1× B200 | 3,410 | 131 | 2,988.50 | 115.81 | 111.98 | 286.90 ms |
| [`sglang/agg-gb200-mtp/deploy.yaml`](../sglang/agg-gb200-mtp/deploy.yaml) | NVFP4 | 1× GB200 | 3,410 | 131 | 3,035.98 | 114.03 | 111.47 | 156.07 ms |
| TBD | TBD | 1× H200 | TBD | TBD | TBD | TBD | TBD | TBD |

The MTP results use three speculative tokens with a synthetic acceptance
length of 3.3153. This value was calculated for the coding workload in
SpeedBench. The full trace produced 131 expected context-length errors.

When benchmarking SGLang with synthetic acceptance, set
`SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` together with
`SGLANG_SIMULATE_ACC_LEN` and `SGLANG_SIMULATE_ACC_METHOD`. SGLang 0.5.16 uses
fixed token ID 100 when the token mode is omitted. That token decodes to the
Unicode replacement character for the Qwen tokenizer, so Dynamo buffers the
generated text until the response finishes and AIPerf cannot calculate
inter-token latency or output throughput per user.
