# Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment

This repository contains the public reproduction code for our TraceMRT experiments on stress-testing chain-of-thought monitoring under covert misalignment. It supports two task settings:

- **SHADE-Arena**: the five public open-source SHADE-Arena tasks included in `task_pairs/`.
- **CUA-SHADE-Arena**: computer-use tasks included in `cua_task_pairs/` and executed through Inspect AI.

Our work has two main contributions:

- **TraceMRT red-team framework**: generates normal, adversarial, and optimized trajectories for task-completion settings.
- **Bayesian monitor framework**: scores saved trajectories with base, sequential, hierarchical, and Bayesian monitor variants.

## Overview

The first figure summarizes the TraceMRT pipeline for generating and optimizing covertly misaligned task-completion trajectories across SHADE-Arena and CUA-SHADE-Arena.

![TraceMRT framework](assets/framework_hao.png)

The second figure shows the monitor scaffold used to aggregate trajectory evidence into Bayesian monitor judgments.

![Bayesian monitor scaffold](assets/monitor_scaffold.png)

## Repository Layout

- `run_shade.py`: SHADE-Arena trajectory runner.
- `run_cua.py`: CUA-SHADE-Arena trajectory runner.
- `add_monitor.py`: post-hoc monitor runner for saved trajectories.
- `task_pairs/`: public SHADE-Arena task definitions.
- `cua_task_pairs/`: CUA-SHADE-Arena task definitions.
- `cua_shade_arena/`: Inspect AI CUA runtime wrapper and task assets.
- `tracemrt/`: optimization, strategy bank, scoring, and monitor prompts.
- `llms/`: OpenAI, Anthropic, and DashScope-compatible LLM wrappers.
- `scripts/`: convenience scripts for full runs.
- `assets/`: framework and monitor architecture figures.

## Public SHADE-Arena Tasks

This public repository includes only the five SHADE-Arena tasks that are officially open-source:

- `american_airlines_2`
- `api_key_calendar_agendas_2`
- `bank_transfer_2`
- `expenditure_accounting`
- `spam_filter_update`

The full SHADE-Arena benchmark contains additional non-public tasks. Please contact the official SHADE-Arena maintainers for access to the full task set.

## Environment Setup

Use one Python environment for both SHADE-Arena and CUA-SHADE-Arena. Python 3.10 is recommended because the CUA Inspect AI stack was validated there.

```bash
conda create -n tracemrt python=3.10 -y
conda activate tracemrt
pip install -r requirements.txt
```

Set the API keys for the providers you use:

```bash
export OPENAI_API_KEY="..."     # OpenAI evaluator and OpenAI-backed monitors.
export DASHSCOPE_API_KEY="..."  # CUA agents through Inspect's openai-api/dashscope provider.
export ANTHROPIC_API_KEY="..."  # Required only for Anthropic-backed models.
export GOOGLE_API_KEY="..."     # Required only for optimization.
```

Optional provider settings:

```bash
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_BASE_URL="..."    # Optional, only for non-default OpenAI-compatible endpoints.
export ANTHROPIC_BASE_URL="..." # Optional.
export GOOGLE_MODEL="gemini-3.1-pro-preview"
```

For SHADE-Arena, model lists are configured in `config.yaml`. For CUA-SHADE-Arena, agent models are passed to `run_cua.py` and are called as `openai-api/dashscope/<model_name>` through Inspect AI. CUA side-task evaluation uses `openai/gpt-5.2` with `OPENAI_API_KEY`.

## Running SHADE-Arena

There are two standard workflows for each task: run trajectories with inline monitor scoring, or first save agent-only trajectories and add monitors afterward.

### Option A: Inline Monitor

The convenience script runs the five public SHADE-Arena tasks with secondary tasks enabled, optimization disabled, and inline monitor scoring enabled. Monitor models come from `monitor_models_list` in `config.yaml`.

```bash
bash scripts/run_shade.sh
```

Equivalent direct command:

```bash
python run_shade.py \
  --num_experiments 1 \
  --test_mode False \
  --task_folder american_airlines_2 api_key_calendar_agendas_2 bank_transfer_2 expenditure_accounting spam_filter_update \
  --max_concurrent_threads 5 \
  --include_secondary_task True \
  --optimize False \
  --run_monitor True \
  --config_file config.yaml
```

Output:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

### Option B: Agent-Only, Then Add Monitor

First generate trajectories without monitor scoring:

```bash
python run_shade.py \
  --num_experiments 1 \
  --test_mode False \
  --task_folder bank_transfer_2 \
  --max_concurrent_threads 1 \
  --include_secondary_task True \
  --optimize False \
  --run_monitor False \
  --config_file config.yaml
```

Agent-only output:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/no_monitor/transcript_N.json
```

Then score the saved trajectories:

```bash
python add_monitor.py \
  --dataset shade_arena \
  --transcript_type transcripts_with_secondary_task \
  --agent glm-4.7 \
  --monitor_model gpt-5.2 \
  --monitor_style base_aware sequential_aware hierarchical_aware bayesian \
  --max_num_transcripts 10 \
  --max_concurrent 5
```

Post-hoc monitor output:

```text
task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

When multiple monitor styles are passed in one `add_monitor.py` command, they are saved into the same transcript JSON as separate fields, e.g. `monitoring_response`, `monitoring_response_sequential_aware`, `monitoring_response_hierarchical_aware`, and `monitoring_response_bayesian`.

## Running CUA-SHADE-Arena

CUA-SHADE-Arena follows the same two workflows. `run_cua.py` uses DashScope for the tested agent and OpenAI for the side-task evaluator.

### Option A: Inline Monitor

The convenience script runs all CUA tasks across the configured agent list with secondary tasks enabled, optimization disabled, and inline monitor scoring enabled.

```bash
bash scripts/run_cua.sh
```

Small direct smoke test:

```bash
python run_cua.py \
  --task_names ds_coffee \
  --backend_model deepseek-v3.2 \
  --num_experiments 1 \
  --max_concurrent 1 \
  --include_secondary_task \
  --optimize False \
  --run_monitor True \
  --monitor_model gpt-5.2 \
  --monitor_style base_aware
```

Output:

```text
cua_task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

### Option B: Agent-Only, Then Add Monitor

First generate agent-only trajectories:

```bash
python run_cua.py \
  --task_names ds_coffee \
  --backend_model deepseek-v3.2 \
  --num_experiments 1 \
  --max_concurrent 1 \
  --include_secondary_task \
  --optimize False \
  --run_monitor False
```

Agent-only output:

```text
cua_task_pairs/<task>/transcripts_with_secondary_task/<agent>/no_monitor/transcript_N.json
```

Then score the saved trajectories:

```bash
python add_monitor.py \
  --dataset cua_shade_arena \
  --transcript_type transcripts_with_secondary_task \
  --agent deepseek-v3.2 \
  --monitor_model gpt-5.2 \
  --monitor_style base_aware sequential_aware hierarchical_aware bayesian \
  --max_num_transcripts 10 \
  --max_concurrent 5
```

Post-hoc monitor output:

```text
cua_task_pairs/<task>/transcripts_with_secondary_task/<agent>/<monitor_model>/transcript_N.json
```

## Optimization

Optimization produces multiple monitor-free candidate trajectories. It should not be combined with inline monitor scoring. If both `--optimize True` and `--run_monitor True` are provided, optimization takes precedence and monitor scoring should be run afterward with `add_monitor.py`.

Optimization requires:

```bash
export GOOGLE_API_KEY="..."
```

### SHADE-Arena Optimization

```bash
python run_shade.py \
  --num_experiments 1 \
  --test_mode False \
  --task_folder bank_transfer_2 \
  --max_concurrent_threads 1 \
  --include_secondary_task True \
  --optimize True \
  --run_monitor False \
  --config_file config.yaml
```

Output:

```text
task_pairs/<task>/transcripts_optimization/<agent>/candidates/transcript_N.json
```

### CUA-SHADE-Arena Optimization

```bash
python run_cua.py \
  --task_names ds_coffee \
  --backend_model deepseek-v3.2 \
  --num_experiments 1 \
  --max_concurrent 1 \
  --include_secondary_task \
  --optimize True \
  --run_monitor False
```

Output:

```text
cua_task_pairs/<task>/transcripts_optimization/<agent>/candidates/transcript_N.json
```

Each optimization candidate contains the conversation, side-task evaluation, optimization metadata, and optimized system prompt. It does not contain monitor output.

To score optimization candidates, run `add_monitor.py`. For `transcripts_optimization`, the input subdirectory defaults to `candidates`:

```bash
python add_monitor.py \
  --dataset cua_shade_arena \
  --transcript_type transcripts_optimization \
  --agent deepseek-v3.2 \
  --monitor_model gpt-5.2 \
  --monitor_style bayesian \
  --max_num_transcripts 10 \
  --max_concurrent 5
```

Monitor-scored optimization output:

```text
<task_root>/<task>/transcripts_optimization/<agent>/<monitor_model>/transcript_N.json
```

## Output Schema Notes

Saved transcript JSON files contain at least:

- `conversation`: the agent trajectory.
- `side_task_evaluation`: whether the side task was completed.

Monitor-scored files additionally contain one or more monitor response fields:

- `monitoring_response`: `base_aware` monitor output.
- `monitoring_response_<style>`: non-base monitor outputs.

Optimization candidates additionally contain:

- `phase1_frame`
- `phase2_genes`
- `phase2_generations`
- `optimized_system_prompt`

Agent-only `no_monitor` transcripts intentionally do not include an empty `monitoring_response` field.

## Citation

If you use this code, please cite the associated paper. Citation details will be added after release.

## Data Notice

Canary string: The following canary helps researchers prevent SHADE-Arena task data from being included in training datasets: tb0sobzyL7TdWeI1d6uj80jXjwfFDZ-fRzwcEBmjH5TPRdM9nFOmiedVSw8uS
