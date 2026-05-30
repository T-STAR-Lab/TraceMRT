import argparse
import os
import json
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import includes
from inspect_ai.log import read_eval_log

from react_agents import custom_react_agent
from evaluation_solver import two_stage_with_evaluation
from utils import export_transcript_json


def load_dataset_from_file(task_name, prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    user_prompt = obj.get("user_prompt", "")
    system_prompt = obj.get("system_prompt", "")

    dataset = MemoryDataset([
        Sample(id=task_name, input=user_prompt)
    ])

    # dataset = MemoryDataset([
    #     Sample(id=task_name, input="count the number of files in the current directory")
    # ])
    return dataset, system_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--inputs_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--model_max_retries", type=int, default=5)
    parser.add_argument("--model_timeout", type=int, default=120)
    parser.add_argument("--model_attempt_timeout", type=int, default=30)
    parser.add_argument("--side_task", type=str, default=None)

    args = parser.parse_args()

    # CUA agents are served through DashScope's OpenAI-compatible endpoint.
    os.environ.setdefault("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if args.model.startswith(("openai-api/", "openai/", "anthropic/", "google/")):
        model_name = args.model
    else:
        model_name = "openai-api/dashscope/" + args.model

    dataset, system_prompt = load_dataset_from_file(args.task_name, args.prompt_file)

    # ===== Select solver =====
    if args.side_task:
        solver_func = two_stage_with_evaluation(
            system_prompt=system_prompt,
            side_task=args.side_task,
        )
    else:
        solver_func = custom_react_agent(system_prompt)

    task = Task(
        dataset=dataset,
        solver=solver_func,
        scorer=includes(),
        sandbox=("docker", "./cua_shade_arena/compose.yaml"),
    )

    eval_results = eval(
        task,
        model=model_name,
        log_dir=f"./cua_shade_arena/logs/{model_name}",
        max_tokens=args.max_tokens,
        max_retries=args.model_max_retries,
        timeout=args.model_timeout,
        attempt_timeout=args.model_attempt_timeout,
        log_format="json",
        time_limit=3600,
        extra_body={"enable_thinking": True},
    )

    log = read_eval_log(eval_results[0].location, resolve_attachments=True)

    # export_transcript_json prefers the saved stage-1 snapshot.
    transcript = export_transcript_json(log, sample_index=0)

    # Add only one top-level field.
    if args.side_task:
        sample = log.samples[0]
        transcript["side_task_evaluation"] = bool(sample.metadata.get("side_task_evaluation", False))
        evaluator_error = sample.metadata.get("_side_task_evaluator_error")
        if evaluator_error:
            transcript["evaluator_error"] = evaluator_error

    print(json.dumps(transcript, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()