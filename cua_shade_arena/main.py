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
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)

    # ===== side_task 评估器参数=====
    parser.add_argument("--side_task", type=str, default=None)
    parser.add_argument("--evaluator_model", type=str, default="openai/gpt-5.2")
    parser.add_argument("--evaluator_api_key", type=str, default="CGFsVd3kI3Ve9DbWyP6Jz922vi4Y7xazTYHkIp4EKUuDOc9glHsfJQQJ99CAACHYHv6XJ3w3AAAAACOGwAfp")
    parser.add_argument("--evaluator_base_url", type=str, default="https://dongy-mkb6b8up-eastus2.cognitiveservices.azure.com/openai/v1/")

    args = parser.parse_args()

    # ===== 配置被测模型 API =====
    if 'claude' in args.model.lower():
        model_name = 'anthropic/' + args.model
        os.environ["ANTHROPIC_API_KEY"] = args.api_key if args.api_key else ""
        if args.base_url:
            os.environ["ANTHROPIC_BASE_URL"] = args.base_url
    else:
        model_name = 'openai/' + args.model
        os.environ["OPENAI_API_KEY"] = args.api_key if args.api_key else ""
        if args.base_url:
            os.environ["OPENAI_BASE_URL"] = args.base_url

    dataset, system_prompt = load_dataset_from_file(args.task_name, args.prompt_file)

    # ===== 选择 solver =====
    if args.side_task:
        if not args.evaluator_api_key:
            raise ValueError("side_task enabled but --evaluator_api_key is not provided.")
        solver_func = two_stage_with_evaluation(
            system_prompt=system_prompt,
            side_task=args.side_task,
            evaluator_model=args.evaluator_model,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_base_url=args.evaluator_base_url,
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
        log_format="json",
        time_limit=3600,
        extra_body={"enable_thinking": True},
    )

    log = read_eval_log(eval_results[0].location, resolve_attachments=True)

    # 关键：export_transcript_json 会优先用 stage1 快照
    transcript = export_transcript_json(log, sample_index=0)

    # 最外层只追加一个字段
    if args.side_task:
        sample = log.samples[0]
        transcript["side_task_evaluation"] = bool(sample.metadata.get("side_task_evaluation", False))

    print(json.dumps(transcript, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()