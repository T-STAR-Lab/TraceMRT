import asyncio
import glob
import json
import argparse
import logging
import time
import os
import signal
import sys
import traceback
from datetime import datetime
from openai import BadRequestError

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_completion import load_task_definition_and_solution
from utils.analysis_utils import extract_verdict
from utils.llm_utils import get_string_to_llm_dict
from utils.logging_utils import json_conversation_to_string, strip_thinking_tags
from utils.path_utils import resolve_path
from utils.pipeline_utils import PipelineUtils


def log_current_state(signum, frame):
    """Signal handler to print current stack trace of all threads"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== State dump at {timestamp} ===")
    print("Current stack trace:")
    traceback.print_stack(frame)
    print("=== End state dump ===\n")


async def process_single_transcript(
    json_file: str,
    out_dir: str,
    monitor_model: str,
    pipeline_utils,
    monitor_to_llm_dict,
    semaphore,
    monitor_style_list,
):
    async with semaphore:
        print(f"{datetime.now().strftime('%H:%M:%S')} - Starting {json_file}...")

        with open(json_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        conversation = transcript["conversation"]
        monitor_input = json_conversation_to_string(conversation[1:])

        # 逐个 style 跑，同一个 transcript dict 写多个字段
        for monitor_style in monitor_style_list:
            try:
                result, _ = await pipeline_utils.run_monitor(
                    monitor_input,
                    monitor_to_llm_dict[monitor_model],
                    is_already_string=True,
                    monitor_style=monitor_style,
                )

            except BadRequestError as e:
                print(f"BadRequestError on json_file: {json_file}, style={monitor_style}, {e}")
                continue
            except Exception as e:
                print(f"Error on json_file: {json_file}, style={monitor_style}, {e}")
                traceback.print_exc()
                continue

            last_content = result.messages[-1].get("content", "")

            ms = monitor_style.strip().lower()
            if ms == "base_aware":
                transcript["monitoring_response"] = last_content
            else:
                transcript[f"monitoring_response_{ms}"] = last_content

        # 最后只保存一次
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, os.path.basename(json_file))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"{datetime.now().strftime('%H:%M:%S')} - Saved to {output_file}")

async def _process_single_task_transcript_monitor_async(
    task_name,
    task_def,
    transcript_type,
    agent,  # 只处理这个 agent
    monitor_model,
    semaphore,
    string_to_llm_dict,
    combination_idx,
    total_combinations,
    max_num_transcripts,
    monitor_style_list,  # 列表：多个 style 写进同一个 json
    dataset="shade_arena",
    in_subdir=None,  # 指定输入子目录，None 则默认使用 monitor_model 名
):
    combination_id = f"{task_name}/{transcript_type}/{agent}"
    print(f"[START] Combination {combination_idx}/{total_combinations}: {combination_id}")

    try:
        pipeline_utils = PipelineUtils(task_def())

        # dataset 根目录逻辑保持不变
        if dataset == "shade_arena":
            base_root = resolve_path(f"task_pairs/{task_name}/{transcript_type}")
        elif dataset == "cua_shade_arena":
            base_root = resolve_path(f"task_pairs_agentcompany/{task_name}/{transcript_type}")
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}. Expected 'shade_arena' or 'cua_shade_arena'."
            )

        agent_dir = os.path.join(base_root, agent)
        in_dir = os.path.join(agent_dir, in_subdir if in_subdir else monitor_model.split("/")[-1])

        if not os.path.isdir(in_dir):
            print(f"[{combination_id}] input dir not found, skipping: {in_dir}")
            return

        json_files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
        if not json_files:
            print(f"[{combination_id}] No JSON files found in {in_dir}")
            return

        print(f"[{combination_id}] Found {len(json_files)} transcripts in {in_dir}")

        # 限制数量（按 transcript_0..N 的名字取）
        if len(json_files) > max_num_transcripts:
            limited = []
            for i in range(max_num_transcripts):
                name = f"transcript_{i}.json"
                m = [f for f in json_files if f.endswith(name)]
                if m:
                    limited.extend(m)
            if limited:
                json_files = limited
            print(f"[{combination_id}] Limiting to first {max_num_transcripts} transcripts")

        monitor_to_llm_dict = string_to_llm_dict

        # 输出目录：按 monitor_model 分，不按 style 分
        monitor_model_dir = monitor_model.split("/")[-1]
        out_dir = os.path.join(agent_dir, monitor_model_dir)
        os.makedirs(out_dir, exist_ok=True)

        print(
            f"[{combination_id}] Output dir: {out_dir} | "
            f"Monitor styles: {monitor_style_list}"
        )

        # 为每个 json_file 创建一次任务（一次任务内跑多个 style）
        transcript_tasks = []
        for json_file in json_files:
            try:
                t = asyncio.create_task(
                    process_single_transcript(
                        json_file=json_file,
                        out_dir=out_dir,
                        monitor_model=monitor_model,
                        pipeline_utils=pipeline_utils,
                        monitor_to_llm_dict=monitor_to_llm_dict,
                        semaphore=semaphore,
                        monitor_style_list=monitor_style_list,
                    )
                )
                transcript_tasks.append(t)
            except Exception as e:
                print(f"[{combination_id}] Error creating task for {json_file}: {e}")
                traceback.print_exc()

        if transcript_tasks:
            await asyncio.gather(*transcript_tasks)

        print(f"[{combination_id}] Completed processing")

    except Exception as e:
        print(f"[{combination_id}] Error processing combination: {e}")
        traceback.print_exc()

    print(f"[END] Combination {combination_idx}/{total_combinations}: {combination_id}")

async def process_many_transcripts(
    task_name_list,
    task_def_list,
    transcript_type_list,
    agent,
    monitor_model,
    semaphore,
    max_num_transcripts,
    monitor_style_list,
    dataset="shade_arena",
    in_subdir=None,
):
    """
    只保留维度：(task_name, transcript_type, agent)
    默认从 agent/no_monitor 读；optimization transcript 默认从 agent/candidates 读。
    """
    string_to_llm_dict = get_string_to_llm_dict([monitor_model])

    total_combinations = len(task_name_list) * len(transcript_type_list)

    print(f"--- Starting concurrent processing for {total_combinations} combinations ---")
    print(
        f"Tasks: {len(task_name_list)}, "
        f"Transcript types: {len(transcript_type_list)}, "
        f"Agent: {agent}, "
        f"Monitor styles: {monitor_style_list}"
    )

    tasks = []
    combination_idx = 0

    for task_name, task_def in zip(task_name_list, task_def_list):
        for transcript_type in transcript_type_list:
            combination_idx += 1
            task = _process_single_task_transcript_monitor_async(
                task_name=task_name,
                task_def=task_def,
                transcript_type=transcript_type,
                agent=agent,
                monitor_model=monitor_model,
                semaphore=semaphore,
                string_to_llm_dict=string_to_llm_dict,
                combination_idx=combination_idx,
                total_combinations=total_combinations,
                max_num_transcripts=max_num_transcripts,
                monitor_style_list=monitor_style_list,
                dataset=dataset,
                in_subdir=in_subdir,
            )
            tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n--- All {total_combinations} combinations have finished ---")

    failed_combinations = 0
    for i, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            failed_combinations += 1
            print(f"Combination {i} resulted in an error: {result}")

    if failed_combinations > 0:
        print(f"Total failed combinations: {failed_combinations}/{total_combinations}")
    else:
        print("All combinations completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monitors on agent transcripts.")
    parser.add_argument(
        "--monitor_model",
        type=str,
        required=True,
        help="The monitor model to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="shade_arena",
        help="Dataset root: shade_arena or cua_shade_arena.",
    )
    parser.add_argument(
        "--transcript_type",
        nargs="+",
        type=str,
        default=["transcripts_with_secondary_task"],
        help="One or more transcript types to use.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent API requests.",
    )
    parser.add_argument(
        "--max_num_transcripts",
        type=int,
        default=100,
        help="Maximum number of transcripts to process per agent_model per task/transcript_type.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent folder name under transcript_type, e.g. deepseek-v3.2 / glm-4.6 / kimi-k2-thinking / qwen3-235b-a22b-thinking-2507",
    )

    parser.add_argument(
        "--monitor_style",
        nargs="+",
        type=str,
        default=["base_aware", "sequential_aware", "hierarchical_aware"],
        help="One or more monitor styles to run (default: base_aware sequential_aware hierarchical_aware)",
    )

    parser.add_argument(
        "--in_subdir",
        type=str,
        default=None,
        help="Input subdirectory under agent dir. Defaults to candidates for transcripts_optimization, otherwise no_monitor.",
    )

    args = parser.parse_args()

    if not args.transcript_type:
        print("Error: at least one --transcript_type is required.")
        sys.exit(1)

    allowed_monitor_styles = {
        "base_aware", "base_unaware", "base_omniscient",
        "sequential_aware", "sequential_unaware", "sequential_omniscient",
        "hierarchical_aware", "hierarchical_unaware", "hierarchical_omniscient",
        "bayesian"
    }

    monitor_style_list = []
    for s in (args.monitor_style or []):
        if s is None:
            continue
        ms = s.strip().lower()
        if ms not in allowed_monitor_styles:
            raise ValueError(
                f"Invalid monitor_style: {s}. Expected one of: {', '.join(sorted(allowed_monitor_styles))}"
            )
        monitor_style_list.append(ms)

    if not monitor_style_list:
        raise ValueError("monitor_style list is empty after parsing; please provide at least one style.")


    # 注册信号
    signal.signal(signal.SIGQUIT, log_current_state)
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(1))
    print("Press Ctrl+\\ to see current execution state")
    print("Press Ctrl+C to force exit")

    # 决定并发度
    if args.max_concurrent is not None:
        MAX_CONCURRENT_REQUESTS = args.max_concurrent
    else:
        MAX_CONCURRENT_REQUESTS = 20

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    dataset_key = (args.dataset or "").strip().lower()

    if dataset_key == "shade_arena":
        task_pairs_root = "task_pairs"
    elif dataset_key == "cua_shade_arena":
        task_pairs_root = "task_pairs_agentcompany"
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. Expected 'shade_arena' or 'cua_shade_arena'."
        )
    if args.in_subdir is None:
        args.in_subdir = "candidates" if any("optimization" in t for t in args.transcript_type) else "no_monitor"


    # 在 task_pairs_root 下自动发现有哪些任务有对应 transcript_type[0] 目录
    task_pairs_dir = resolve_path(task_pairs_root)
    if not os.path.exists(task_pairs_dir):
        print(f"Error: {task_pairs_root} directory not found at {task_pairs_dir}")
        sys.exit(1)

    first_type = args.transcript_type[0]
    task_dirs = [
        d
        for d in os.listdir(task_pairs_dir)
        if os.path.isdir(resolve_path(os.path.join(task_pairs_root, d)))
        and not d.startswith("__")
        and os.path.isdir(resolve_path(os.path.join(task_pairs_root, d, first_type)))
    ]

    task_name_list = sorted(task_dirs)
    if not task_name_list:
        print(
            f"No tasks found under {task_pairs_root} with transcript_type '{first_type}'. "
            f"Nothing to do."
        )
        sys.exit(0)

    task_def_list = [
        load_task_definition_and_solution(
            str(resolve_path(f"{task_pairs_root}/{task_name}"))
        )[0]
        for task_name in task_name_list
    ]

    try:
        asyncio.run(
            process_many_transcripts(
                task_name_list=task_name_list,
                task_def_list=task_def_list,
                transcript_type_list=args.transcript_type,
                agent=args.agent,
                monitor_model=args.monitor_model,
                semaphore=semaphore,
                max_num_transcripts=args.max_num_transcripts,
                monitor_style_list=monitor_style_list,
                dataset=args.dataset,
                in_subdir=args.in_subdir,
            )
        )

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
