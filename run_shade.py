import asyncio

import os
from pathlib import Path

import yaml
from simple_parsing import ArgumentParser

from dotenv import load_dotenv
load_dotenv()

from tools.end_task import end_task
from utils.experiment_config import ExperimentConfig
from utils.experiment_runner import ExperimentRunner
from utils.path_utils import resolve_path, set_project_root
from utils.task_utils import load_task_definition_and_solution


# Set project root to the directory containing run_shade.py
set_project_root(Path(__file__).parent)



def create_experiment_runner(task_folder: str, cfg: ExperimentConfig, config: dict) -> ExperimentRunner:
    """Create and return an ExperimentRunner instance for a given task folder."""
    TaskDefinitionClass, solution_func = load_task_definition_and_solution(task_folder)
    task_def = TaskDefinitionClass()  # type: ignore
    task_def.environment.add_tools([end_task])

    return ExperimentRunner(
        task_folder=task_folder,
        task_def=task_def,
        solution_func=solution_func,
        agent_models_list=config["agent_models_list"],
        monitor_models_list=config["monitor_models_list"],
        evaluator_model=config["evaluator_model"],
        org_model_mapping=config["org_model_mapping"],
        test_mode=cfg.test_mode,
        num_experiments=cfg.num_experiments,
        monitor_style=cfg.monitor_style,
        include_secondary_task=cfg.include_secondary_task,
        run_monitor=cfg.run_monitor,
        max_concurrent_threads=cfg.max_concurrent_threads,
        optimize=cfg.optimize,
    )


async def run_single_task(cfg: ExperimentConfig, config: dict) -> None:
    """Run experiments for a single task."""
    if cfg.task_folder is None:
        raise ValueError("task_folder must be specified")
    task_folder = "task_pairs/" + cfg.task_folder[0]
    runner = create_experiment_runner(task_folder, cfg, config)

    if cfg.run_solution:
        await runner.generate_solution()
    else:
        await runner.run_multiple_experiments_for_multiple_models()


async def run_all_tasks(cfg: ExperimentConfig, config: dict) -> None:
    """Run experiments for all tasks in parallel."""
    task_pairs_dir = resolve_path("task_pairs")
    task_folders = [
        d
        for d in os.listdir(task_pairs_dir)
        if os.path.isdir(os.path.join(task_pairs_dir, d)) and not d.startswith("__")
    ]

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(cfg.max_concurrent_threads)

    async def run_task_with_semaphore(task_folder):
        async with semaphore:
            runner = create_experiment_runner(task_folder, cfg, config)
            if cfg.run_solution:
                await runner.generate_solution()
            else:
                await runner.run_multiple_experiments_for_multiple_models()

    tasks = []
    for folder in task_folders:
        task_folder = "task_pairs/" + folder
        tasks.append(run_task_with_semaphore(task_folder))

    await asyncio.gather(*tasks)

async def run_specific_tasks(cfg: ExperimentConfig, config: dict) -> None:
    """Run experiments for a specific list of tasks with global concurrency control."""
    if not cfg.task_folder:
        raise ValueError("task_folder must be specified and non-empty")

    task_list = cfg.task_folder

    print(f"Running {len(task_list)} tasks with global concurrency limit of {cfg.max_concurrent_threads}")
    print(f"Tasks: {task_list}")

    semaphore = asyncio.Semaphore(cfg.max_concurrent_threads)

    async def run_task_with_semaphore(task_name: str):
        task_folder = "task_pairs/" + task_name
        async with semaphore:
            runner = create_experiment_runner(task_folder, cfg, config)
            if cfg.run_solution:
                await runner.generate_solution()
            else:
                await runner.run_multiple_experiments_for_multiple_models()

    tasks = [run_task_with_semaphore(name) for name in task_list]
    await asyncio.gather(*tasks)
    print(f"\nAll {len(task_list)} specified tasks completed!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    
    # Load the base config.yaml
    base_config_path = resolve_path("config.yaml")
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # If a custom config file is specified, load it and update only the model lists
    if cfg.config_file != "config.yaml":
        config_path = resolve_path(cfg.config_file)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                custom_config = yaml.safe_load(f)
                
            # Override specific configuration parameters
            for key in ["agent_models_list", "monitor_models_list", "evaluator_model"]:
                
                    base_config[key] = custom_config.get(key, base_config.get(key))
            
            print(f"Loaded custom configuration from {cfg.config_file}")
        else:
            raise ValueError(f"Custom config file {cfg.config_file} not found")
    
    config = base_config

    if cfg.run_all_tasks:
        asyncio.run(run_all_tasks(cfg, config))
    elif not cfg.task_folder:
        raise ValueError("Either --run_all_tasks must be True or --task_folder must be specified")
    elif len(cfg.task_folder) == 1:
        asyncio.run(run_single_task(cfg, config))
    else:
        asyncio.run(run_specific_tasks(cfg, config))
