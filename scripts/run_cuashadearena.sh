#!/usr/bin/env bash

set -euo pipefail

agents=(
    "kimi-k2-thinking"
    "qwen3-235b-a22b-thinking-2507"
    "glm-4.7"
    "deepseek-v3.2"
)

tasks=(
    "admin_make_spreadsheet"
    "admin_mass_forms_filling"
    "admin_remove_pages"
    "ds_coffee"
    "ds_find_meeting_spreadsheet"
    "ds_fix_table_values_and_missing_answers"
    "ds_format_excel_sheets"
    "finance_budget_variance"
    "finance_check_attendance_payroll"
    "finance_expense_validation"
    "finance_invoice_matching"
    "hr_check_attendance_multiple_days"
    "hr_populate_salary_increase_memo"
    "hr_salary_analysis"
    "research_answer_questions"
)

for task in "${tasks[@]}"
do
    echo "----------------------------------------------------"
    echo "[Current Task] Starting inspection for task: $task"
    echo "----------------------------------------------------"

    for agent in "${agents[@]}"
    do
        echo "[Starting] Model: $agent for task: $task"
        python task_completion_inspect.py \
          --task_names "$task" \
          --backend_model "$agent" \
          --num_experiments 1 \
          --max_concurrent 5 \
          --api_key "${DASHSCOPE_API_KEY}" \
          --base_url "${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
          --include_secondary_task \
          --optimize False \
          --run_monitor True \
          --monitor_model gpt-5.2 \
          --monitor_style base_aware &
    done

    wait || true

    echo "[Finished] Completed inspection for task: $task (all models)"
    echo ""

done

echo "All tasks processed successfully."
