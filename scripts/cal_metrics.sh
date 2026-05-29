#!/bin/bash

python cal_metrics.py \
  --task_pairs_root ./task_pairs \
  --transcripts_type transcripts_optimization \
  --agents deepseek-v3.2 glm-4.7 kimi-k2-thinking qwen3-235b-a22b-thinking-2507 \
  --monitor AgentDoG-Qwen3-4B \