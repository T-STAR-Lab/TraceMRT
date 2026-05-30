#!/usr/bin/env bash

set -euo pipefail

TASKS=("american_airlines_2" "api_key_calendar_agendas_2" "bank_transfer_2" "expenditure_accounting" "spam_filter_update")

python run_shade.py \
    --num_experiments 1 \
    --test_mode False \
    --task_folder "${TASKS[@]}" \
    --max_concurrent_threads 5 \
    --include_secondary_task True \
    --optimize False \
    --run_monitor True \
    --config_file "config.yaml"

echo "All SHADE-Arena runs completed."