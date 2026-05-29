#!/bin/bash

agents=(
    "kimi-k2-thinking"
    "glm-4.7"
    "deepseek-v3.2"
    "qwen3-235b-a22b-thinking-2507"
)


for agent in "${agents[@]}"
do
    echo "----------------------------------------------------"
    echo "[Current Task] Starting monitor for agent: $agent"
    echo "----------------------------------------------------"

    python add_monitor.py \
      --monitor_model gpt-5.2 \
      --dataset shade_arena \
      --transcript_type transcripts_with_secondary_task \
      --agent "$agent" \
      --monitor_style bayesian \
      --in_subdir candidates \
      --max_concurrent 10 \
      --max_num_transcripts 200

    echo "[Finished] Completed monitor task for agent: $agent"
    echo ""
    
done

echo "All agents processed successfully."