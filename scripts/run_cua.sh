#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "Set DASHSCOPE_API_KEY before running this script." >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY before running this script." >&2
    exit 1
fi

export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

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

common_args=(
    --task_names "${tasks[@]}"
    --num_experiments 1
    --max_concurrent 5
    --include_secondary_task
    --optimize False
    --run_monitor True
    --monitor_model gpt-5.2
    --monitor_style base_aware
)

for agent in "${agents[@]}"; do
    echo "[Starting] CUA-SHADE-Arena agent: ${agent}"
    python run_cua.py \
        --backend_model "${agent}" \
        "${common_args[@]}" &
done

wait

echo "All CUA-SHADE-Arena runs completed."
