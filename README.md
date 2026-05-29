# TraceMRT: SHADE-Arena and CUA-SHADE-Arena

This repository contains the public reproduction code for two TraceMRT task settings:

- **SHADE-Arena**: the five public open-source SHADE-Arena tasks included under `task_pairs/`.
- **CUA-SHADE-Arena**: computer-use tasks included under `task_pairs_agentcompany/` and run through `task_completion_inspect.py`.

The code supports normal agent trajectory generation, optional inline monitor scoring, and optimization-based candidate generation. Optimization candidates are monitor-free by design and should be scored afterward with `add_monitor.py`.

## Public SHADE-Arena Tasks

The public SHADE-Arena task set in this repo contains:

- `american_airlines_2`
- `api_key_calendar_agendas_2`
- `bank_transfer_2`
- `expenditure_accounting`
- `spam_filter_update`

## Repository Layout

- `task_completion.py`: SHADE-Arena runner.
- `task_completion_inspect.py`: CUA-SHADE-Arena runner.
- `add_monitor.py`: post-hoc monitor runner for saved transcripts.
- `task_pairs/`: SHADE-Arena task definitions.
- `task_pairs_agentcompany/`: CUA-SHADE-Arena task definitions.
- `task_evaluation/`: SHADE-Arena task evaluators.
- `cua_shade_arena/`: CUA runtime wrapper and task assets.
- `tracemrt/`: optimization, scoring, strategy, and monitor prompt utilities.
- `llms/`: LLM provider wrappers.
- `tools/`, `environment_classes/`, `environments/`, `universe_related_stuff/`: mock tools and environment data.
- `scripts/`: convenience scripts for full runs.

## Setup

Use Python 3.11. A typical setup is:

```bash
conda create -n shade-arena python=3.11 -y
conda activate shade-arena
pip install -r requirements.txt
```

Set the API keys needed by the models you use:

```bash
export OPENAI_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export OPTIMIZER_API_KEY="..."  # Required only for optimization.
```

Optional provider settings:

```bash
export OPENAI_BASE_URL="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPTIMIZER_MODEL="gemini-3-flash-preview"
```

`config.yaml` controls the SHADE-Arena agent models, monitor models, evaluator model, and provider mapping. CUA-SHADE-Arena scripts pass the backend and monitor models directly as CLI arguments.

## Running SHADE-Arena

The default SHADE script runs all five public tasks with secondary tasks enabled, optimization disabled, and inline monitor scoring enabled. Monitor models come from `monitor_models_list` in `config.yaml`.

```bash
bash scripts/run_shadearena.sh
```

Equivalent direct command:

```bash
python task_completion.py \
  --num_experiments 1 \
  --test_mode False \
  --task_folder american_airlines_2 api_key_calendar_agendas_2 bank_transfer_2 expenditure_accounting spam_filter_update \
  --max_concurrent_threads 5 \
  --include_secondary_task True \
  --optimize False \
  --run_monitor True \
  --config_file config.yaml
```

Inline monitor outputs are saved under:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

If `--run_monitor False`, agent-only outputs are saved under:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/no_monitor/transcript_N.json
```

## Running CUA-SHADE-Arena

The default CUA script runs all CUA tasks across the configured agent list with secondary tasks enabled, optimization disabled, and inline monitor scoring enabled.

```bash
bash scripts/run_cuashadearena.sh
```

Equivalent direct command for a small smoke test:

```bash
python task_completion_inspect.py \
  --task_names ds_coffee \
  --backend_model deepseek-v3.2 \
  --num_experiments 1 \
  --max_concurrent 1 \
  --api_key "$DASHSCOPE_API_KEY" \
  --base_url "${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
  --include_secondary_task \
  --optimize False \
  --run_monitor True \
  --monitor_model gpt-5.2 \
  --monitor_style base_aware
```

Inline monitor outputs are saved under:

```text
task_pairs_agentcompany/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

If `--run_monitor False`, CUA agent-only outputs are saved under:

```text
task_pairs_agentcompany/<task>/transcripts_with_secondary_task/<agent>/no_monitor/transcript_N.json
```

## Optimization Runs

Optimization produces multiple monitor-free candidate trajectories. It should not be combined with inline monitor scoring.

For SHADE-Arena:

```bash
python task_completion.py \
  --num_experiments 1 \
  --test_mode False \
  --task_folder bank_transfer_2 \
  --max_concurrent_threads 1 \
  --include_secondary_task True \
  --optimize True \
  --run_monitor False \
  --config_file config.yaml
```

For CUA-SHADE-Arena:

```bash
python task_completion_inspect.py \
  --task_names ds_coffee \
  --backend_model deepseek-v3.2 \
  --num_experiments 1 \
  --max_concurrent 1 \
  --api_key "$DASHSCOPE_API_KEY" \
  --base_url "${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
  --include_secondary_task \
  --optimize True \
  --run_monitor False
```

Optimization candidates are saved under:

```text
task_pairs/<task>/transcripts_optimization/<agent>/candidates/transcript_N.json
task_pairs_agentcompany/<task>/transcripts_optimization/<agent>/candidates/transcript_N.json
```

Each candidate contains the conversation, side-task evaluation, optimization metadata, and optimized system prompt. It does not contain monitor output.

## Adding Monitors Afterward

Use `add_monitor.py` to score existing agent-only trajectories or optimization candidates. For normal transcript types, the input defaults to `no_monitor`. For `transcripts_optimization`, the input defaults to `candidates`.

SHADE-Arena example:

```bash
python add_monitor.py \
  --dataset shade_arena \
  --transcript_type transcripts_with_secondary_task \
  --agent glm-4.7 \
  --monitor_model gpt-5.2 \
  --monitor_style base_aware \
  --max_num_transcripts 10 \
  --max_concurrent 5
```

CUA-SHADE-Arena optimization example:

```bash
python add_monitor.py \
  --dataset cua_shade_arena \
  --transcript_type transcripts_optimization \
  --agent deepseek-v3.2 \
  --monitor_model gpt-5.2 \
  --monitor_style bayesian_improve \
  --in_subdir candidates \
  --max_num_transcripts 10 \
  --max_concurrent 5
```

Post-hoc monitor outputs are saved as transcript copies under:

```text
<task_root>/<task>/<transcript_type>/<agent>/<monitor_model>/transcript_N.json
```

When multiple monitor styles are passed in one command, they are stored in the same JSON file as separate fields, for example `monitoring_response`, `monitoring_response_sequential_aware`, and `_monitor_stats`.

## Output Summary

Normal SHADE-Arena run with inline monitor:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

Normal CUA-SHADE-Arena run with inline monitor:

```text
task_pairs_agentcompany/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

Agent-only run:

```text
<task_root>/<task>/<transcript_type>/<agent>/no_monitor/transcript_N.json
```

Optimization run:

```text
<task_root>/<task>/transcripts_optimization/<agent>/candidates/transcript_N.json
```

## Notes

- Generated transcripts, optimization outputs, caches, logs, and local secret files are ignored by `.gitignore`.
- `no_monitor` transcripts do not include an empty `monitoring_response` field.
- If `--optimize True` and `--run_monitor True` are combined, optimization takes precedence and monitor scoring should be run afterward with `add_monitor.py`.
- For CUA-SHADE-Arena, `--run_monitor True` requires `--monitor_model`.

Canary string: The following canary helps researchers prevent SHADE-Arena task data from being included in training datasets: tb0sobzyL7TdWeI1d6uj80jXjwfFDZ-fRzwcEBmjH5TPRdM9nFOmiedVSw8uS
