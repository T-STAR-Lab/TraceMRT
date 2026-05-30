#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_shade import load_task_definition_and_solution
from utils.llm_utils import get_string_to_llm_dict
from utils.logging_utils import json_conversation_to_string
from utils.path_utils import resolve_path
from utils.pipeline_utils import PipelineUtils

from tracemrt.optimization import phase_one, phase_two, GAConfig


# --- Configuration & Paths ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
INSPECT_DIR = os.path.join(REPO_ROOT, "cua_shade_arena")
TASKS_DIR = os.path.join(REPO_ROOT, "cua_task_pairs")
FORMATS_PATH = os.path.join(INSPECT_DIR, "formats.json")
TAC_OPENHANDS_DIR = os.path.join(INSPECT_DIR, "tac-openhands")
INSPECT_RUNNER_PATH = os.path.join(REPO_ROOT, "cua_shade_arena", "main.py")
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"



# ---------------------------
# Task/template utilities
# ---------------------------

def load_task_definition(task_name: str) -> Tuple[str, str]:
    """Load primary and secondary tasks from JSON definition."""
    json_path = os.path.join(TASKS_DIR, task_name, "task.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Using empty tasks.")
        return "", ""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("primary_task", "") or "", data.get("secondary_task", "") or ""
    except Exception as e:
        print(f"Warning: Failed to parse {json_path}: {e}")
        return "", ""

def load_side_task(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    side_task = obj.get("secondary_task", "")
    return side_task

def load_format_template(format_name: str) -> str:
    """Load template string for the given format_name."""
    try:
        if os.path.exists(FORMATS_PATH):
            with open(FORMATS_PATH, "r", encoding="utf-8") as f:
                formats_map = json.load(f)
            if isinstance(formats_map, dict) and format_name in formats_map:
                template = formats_map[format_name]
                if isinstance(template, str) and template.strip():
                    return template
    except Exception as exc:
        print(f"Warning: Failed to load formats.json: {exc}")

    return "{primary_task}\n\n{secondary_task}"

def render_brief(primary: str, secondary: str, template: str) -> dict:
    """Render a brief dict from primary task, secondary task, and template.

    Returns a dict with keys: system_prompt, user_prompt, secondary_task.
    """
    if secondary:
        system_prompt = template.replace(
            "{secondary_task}", secondary.strip()
        ).strip()
    else:
        system_prompt = template.strip()

    user_prompt = (primary or "").strip()

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "secondary_task": (secondary or "").strip(),
    }


def brief_to_json(brief: dict) -> str:
    """Serialize a brief dict to JSON string."""
    return json.dumps(brief, indent=2, ensure_ascii=False)

def safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def extract_conversation_json(stdout: str) -> dict:
    """Extract the first JSON object that contains a top-level 'conversation' field from stdout."""
    if not stdout:
        raise ValueError("stdout is empty")

    marker = '{"conversation":'
    start = stdout.find(marker)

    if start == -1:
        # fallback: search for the key, then backtrack to nearest '{'
        marker2 = '"conversation":'
        key_pos = stdout.find(marker2)
        if key_pos == -1:
            raise ValueError('Cannot find JSON marker {"conversation": in stdout')
        obj_start = stdout.rfind("{", 0, key_pos)
        if obj_start == -1:
            raise ValueError("Found 'conversation' key but cannot locate '{' before it")
        start = obj_start

    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(stdout[start:])
    if not isinstance(obj, dict) or "conversation" not in obj:
        raise ValueError("Decoded JSON does not look like expected transcript object")
    return obj


# ---------------------------
# Transcript indexing
# ---------------------------

def next_transcript_index(log_path: str) -> int:
    """Return next available transcript index (transcript_{N}.json) under log_path."""
    p = Path(log_path)
    p.mkdir(parents=True, exist_ok=True)

    max_idx = -1
    pat = re.compile(r"^transcript_(\d+)\.json$")

    for f in p.iterdir():
        if f.is_file():
            m = pat.match(f.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

    return max_idx + 1


def monitor_dir_name(monitor_model: Optional[str]) -> str:
    """Return the stable transcript subdirectory for a monitor setting."""
    return monitor_model.split("/")[-1] if monitor_model else "no_monitor"


def ensure_dashscope_env() -> None:
    os.environ.setdefault("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL)


def transcript_type(args: argparse.Namespace) -> str:
    if not args.include_secondary_task:
        return "transcripts_no_secondary_task"
    if args.optimize:
        return "transcripts_optimization"
    return "transcripts_with_secondary_task"


def transcript_dir(task_name: str, args: argparse.Namespace) -> str:
    return os.path.join(
        TASKS_DIR,
        task_name,
        transcript_type(args),
        args.backend_model.split("/")[-1],
        monitor_dir_name(args.monitor_model),
    )


def make_error_transcript(*, rc: Optional[int], stderr: str = "", parse_error: Optional[str] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"rc": rc}
    if stderr:
        error["stderr"] = stderr
    if parse_error:
        error["parse_error"] = parse_error
    return {
        "conversation": [],
        "experiment_metadata": {"side_task_evaluation": False},
        "error": error,
    }


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_task_names(args: argparse.Namespace) -> Optional[List[str]]:
    return args.task_names or getattr(args, "task_names_alt", None)


def allocate_transcript_index(log_path: str, counters: Dict[str, int]) -> int:
    if log_path not in counters:
        counters[log_path] = next_transcript_index(log_path)
    idx = counters[log_path]
    counters[log_path] += 1
    return idx


# ---------------------------
# Light adapter for GA scoring
# ---------------------------

@dataclass
class TranscriptLike:
    """Minimal object to satisfy your existing scoring helpers.

    Your scoring functions access:
    - transcript.monitor_results
    - transcript.experiment_metadata

    We keep raw transcript dict in .raw for debugging if needed.
    """
    raw: Dict[str, Any]
    monitor_results: List[Dict[str, Any]]
    experiment_metadata: Dict[str, Any]


# ---------------------------
# Inspect runner wrapper
# ---------------------------

class InspectRunner:
    def __init__(
        self,
        *,
        inspect_runner_path: str,
        inputs_dir: str,
        backend_model: str,
        max_tokens: int,
        monitor_model: Optional[str] = None,
        monitor_style: str = "base_aware",
    ):
        self.inspect_runner_path = inspect_runner_path
        self.inputs_dir = inputs_dir
        self.backend_model = backend_model
        self.max_tokens = int(max_tokens)
        self.monitor_model = monitor_model
        self.monitor_style = (monitor_style or "base_aware").strip().lower()

        self.monitor_to_llm_dict = None
        if self.monitor_model:
            self.monitor_to_llm_dict = get_string_to_llm_dict([self.monitor_model])

        # cache per task
        self._pipeline_cache: Dict[str, PipelineUtils] = {}

    def _get_pipeline_utils(self, task_name: str) -> PipelineUtils:
        if task_name in self._pipeline_cache:
            return self._pipeline_cache[task_name]

        task_root = resolve_path(f"cua_task_pairs/{task_name}")
        task_def = load_task_definition_and_solution(str(task_root))[0]
        pu = PipelineUtils(task_def())
        self._pipeline_cache[task_name] = pu
        return pu

    def build_command(self, *, task_name: str, prompt_file: str, side_task: Optional[str] = "") -> List[str]:
        cmd = [
            sys.executable,
            self.inspect_runner_path,
            "--task_name",
            task_name,
            "--prompt_file",
            prompt_file,
            "--inputs_dir",
            self.inputs_dir,
            "--model",
            self.backend_model,
            "--max_tokens",
            str(self.max_tokens),
        ]
        if side_task:
            cmd.extend(["--side_task", side_task])
        return cmd

    async def run_once(
        self,
        *,
        task_name: str,
        prompt_file: str,
        sem: asyncio.Semaphore,
        side_task: Optional[str] = "",
    ) -> Dict[str, Any]:
        """Run inspect runner once and return transcript dict (conversation + optional error)."""
        cmd = self.build_command(task_name=task_name, prompt_file=prompt_file, side_task=side_task)

        async with sem:
            p = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await p.communicate()

        out = out_b.decode(errors="ignore")
        err = err_b.decode(errors="ignore")

        if p.returncode != 0:
            return make_error_transcript(rc=p.returncode, stderr=err)

        transcript_obj = extract_conversation_json(out)
        transcript_obj.setdefault("experiment_metadata", {})
        transcript_obj["experiment_metadata"].setdefault("side_task_evaluation", transcript_obj.get("side_task_evaluation", False))
        return transcript_obj

    async def add_monitor(
        self,
        *,
        task_name: str,
        transcript_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optionally run monitor and write monitoring_response + monitor_results into transcript_obj."""
        if not self.monitor_model or not self.monitor_to_llm_dict:
            return transcript_obj

        try:
            pipeline_utils = self._get_pipeline_utils(task_name)
            conversation = transcript_obj.get("conversation", [])
            monitor_input = json_conversation_to_string(conversation[1:])  # skip system

            result, _ = await pipeline_utils.run_monitor(
                monitor_input,
                self.monitor_to_llm_dict[self.monitor_model],
                is_already_string=True,
                monitor_style=self.monitor_style,
            )

            last_content = result.messages[-1].get("content", "") or ""
            if self.monitor_style == "base_aware":
                transcript_obj["monitoring_response"] = last_content
            else:
                transcript_obj[f"monitoring_response_{self.monitor_style}"] = last_content

            # For GA scoring compatibility: store monitor outputs in a uniform structure
            transcript_obj.setdefault("monitor_results", [])
            transcript_obj["monitor_results"].append({"messages": [{"content": last_content}]})

        except Exception as e:
            transcript_obj[f"monitor_error_{self.monitor_style}"] = str(e)

        return transcript_obj


# ---------------------------
# Subprocess concurrency (final runs)
# ---------------------------

async def run_one(job: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        p = await asyncio.create_subprocess_exec(
            *job["cmd"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await p.communicate()

    return {
        "task_name": job["task_name"],
        "cnt": job["cnt"],
        "rc": p.returncode,
        "stdout": out_b.decode(errors="ignore"),
        "stderr": err_b.decode(errors="ignore"),
    }


async def run_all(jobs: List[Dict[str, Any]], max_concurrent: int) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(int(max_concurrent))
    tasks = [asyncio.create_task(run_one(job, sem)) for job in jobs]
    return await asyncio.gather(*tasks)


@dataclass
class OptimizationResult:
    """Result of the optimization process."""
    system_prompt: str
    phase1_frame: Optional[str] = None
    phase2_genes: Optional[Dict[str, int]] = None
    phase2_generations: int = 0
    best_transcript: Optional[Dict[str, Any]] = None


# ---------------------------
# Optimization hook (phase1/phase2)
# ---------------------------

async def optimize_secondary(
    *,
    primary: str,
    secondary: str,
    system_prompt: str,
    enable: bool,
    agent_model: str,
    extra_args: Optional[dict] = None,

    task_name: str,
    format_template: str,
    briefs_dir: str,

    runner: InspectRunner,
    run_sem: asyncio.Semaphore,
) -> List[OptimizationResult]:
    """Return optimization results including system_prompt and metadata.

    - Does NOT save candidate transcripts.
    - Uses InspectRunner to evaluate each GA candidate.
    - Attaches monitor outputs into a TranscriptLike object so your scoring works.
    """
    if not enable:
        return [OptimizationResult(system_prompt=system_prompt)]

    extra_args = extra_args or {}

    # ===== Phase 1 =====
    s1, frame_name = await phase_one(
        main_task=primary,
        side_task=secondary,
        system_prompt=system_prompt,
        agent_model=agent_model,
        extra_args=extra_args,
    )
    base_system_prompt = (s1 or "").strip() or system_prompt

    # ===== Phase 2: evaluate_fn =====
    async def evaluate_fn(candidate_system_prompt: str) -> Any:
        os.makedirs(briefs_dir, exist_ok=True)

        # keep briefs distinct but do not write json candidates
        cand_brief_path = os.path.join(
            briefs_dir,
            f"{task_name}_brief_ga_{abs(hash(candidate_system_prompt)) % 10**12}.txt",
        )
        # Construct brief directly: use optimized system_prompt but keep original secondary
        cand_brief = {
            "system_prompt": candidate_system_prompt,
            "user_prompt": primary.strip(),
            "secondary_task": secondary.strip(),
        }
        with open(cand_brief_path, "w", encoding="utf-8") as f:
            f.write(brief_to_json(cand_brief))

        transcript_obj = await runner.run_once(
            task_name=task_name,
            prompt_file=cand_brief_path,
            sem=run_sem,
            side_task=secondary,
        )

        exp_meta = transcript_obj.get("experiment_metadata", {}) or {}
        return TranscriptLike(raw=transcript_obj, monitor_results=[], experiment_metadata=exp_meta)

    cfg = GAConfig(
        population_size=10,
        generations=5,
        elite_k=2,
        tournament_k=3,
        mutation_prob=0.2,
        max_concurrency=10,
        lam=1.0,                # retained for compatibility; optimization candidates are monitor-free
    )

    all_results, gen_cnt = await phase_two(
        base_system_prompt=base_system_prompt,
        evaluate_fn=evaluate_fn,
        cfg=cfg,
        agent_model=agent_model,
        extra_args=extra_args,
        main_task=primary,
        side_task=secondary,
    )

    # Build OptimizationResult records for every result.
    optimization_results: List[OptimizationResult] = []
    for result in all_results:
        result_system_prompt = (result.system_prompt or "").strip() or system_prompt
        result_transcript = None
        if result.transcript and hasattr(result.transcript, 'raw'):
            result_transcript = result.transcript.raw

        optimization_results.append(OptimizationResult(
            system_prompt=result_system_prompt,
            phase1_frame=frame_name,
            phase2_genes={"o_idx": result.individual.o_idx, "p_idx": result.individual.p_idx},
            phase2_generations=gen_cnt,
            best_transcript=result_transcript,
        ))

    return optimization_results


# ---------------------------
# CLI
# ---------------------------

def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUA-SHADE-Arena tasks with Inspect AI")
    parser.add_argument("--include_secondary_task", action="store_true", help="Include the secondary task in the system prompt")
    parser.add_argument("--task_names", nargs="+", help="Space-separated list of task names")
    parser.add_argument("--task_folder", nargs="+", dest="task_names_alt", help="Alias for --task_names")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of runs per task")
    parser.add_argument("--backend_model", required=True, help="Model identifier (e.g., openai/gpt-4o)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max concurrent Inspect AI runs")

    parser.add_argument("--optimize", type=str2bool, default=False, metavar="True|False", help="Enable phase1/phase2 optimization")
    parser.add_argument("--run_monitor", type=str2bool, default=False, metavar="True|False", help="Run monitor during trajectory generation. Use add_monitor.py instead after optimization.")

    parser.add_argument("--monitor_model", default=None, help="Monitor model to use when --run_monitor True (e.g., gpt-5.2)")
    parser.add_argument("--monitor_style", default="base_aware", help="Monitor style (e.g., base_aware)")

    parser.add_argument("--sandbox_type", default="docker", help="Sandbox type (docker/local)")

    args = parser.parse_args()

    if args.optimize and not args.include_secondary_task:
        raise ValueError("--optimize requires --include_secondary_task")

    if args.optimize and args.run_monitor:
        print("Warning: --run_monitor True is ignored when --optimize True. Run add_monitor.py on candidates after optimization.")
        args.run_monitor = False
        args.monitor_model = None
    elif args.monitor_model and not args.run_monitor:
        print("Warning: --monitor_model was provided, enabling --run_monitor True.")
        args.run_monitor = True

    if args.run_monitor and not args.monitor_model:
        raise ValueError("--run_monitor True requires --monitor_model")

    if not args.run_monitor:
        args.monitor_model = None

    if args.monitor_model:
        allowed_monitor_styles = {
            "base_aware",
            "base_unaware",
            "base_omniscient",
            "sequential_aware",
            "sequential_unaware",
            "sequential_omniscient",
            "hierarchical_aware",
            "hierarchical_unaware",
            "hierarchical_omniscient",
            "bayesian",
        }
        ms = (args.monitor_style or "base_aware").strip().lower()
        if ms not in allowed_monitor_styles:
            raise ValueError(
                f"Invalid monitor_style: {args.monitor_style}. "
                f"Expected one of: {', '.join(sorted(allowed_monitor_styles))}"
            )
        args.monitor_style = ms

    return args


# ---------------------------
# Main
# ---------------------------

async def async_main() -> int:
    args = parse_args()
    ensure_dashscope_env()

    task_names = resolve_task_names(args)
    if not task_names:
        print("Error: You must provide task names via --task_names")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.backend_model.split("/")[-1]
    run_base_dir = os.path.join(INSPECT_DIR, f"run_inspect_{model_short}_{timestamp}")
    briefs_dir = os.path.join(run_base_dir, "briefs")
    os.makedirs(briefs_dir, exist_ok=True)

    print(f"=== Starting CUA-SHADE Run: {timestamp} ===")
    print(f"Experiment Dir: {run_base_dir}")
    print(f"Tasks: {len(task_names)}")
    print(f"Agent model: {args.backend_model}")
    print(f"Transcript type: {transcript_type(args)}")
    if args.monitor_model:
        print(f"Inline monitor: {args.monitor_model} ({args.monitor_style})")

    format_name = "include_side_task" if args.include_secondary_task else "without_side_task"
    format_template = load_format_template(format_name)

    runner = InspectRunner(
        inspect_runner_path=INSPECT_RUNNER_PATH,
        inputs_dir=TAC_OPENHANDS_DIR,
        backend_model=args.backend_model,
        max_tokens=args.max_tokens,
        monitor_model=args.monitor_model,
        monitor_style=args.monitor_style,
    )

    ga_run_sem = asyncio.Semaphore(5)
    task_brief_map: Dict[str, str] = {}

    for task_name in task_names:
        primary, secondary = load_task_definition(task_name)
        brief_content = render_brief(
            primary,
            secondary if args.include_secondary_task else None,
            format_template,
        )

        opt_results: Optional[List[OptimizationResult]] = None
        if args.include_secondary_task and args.optimize:
            opt_results = await optimize_secondary(
                primary=primary,
                secondary=secondary,
                system_prompt=brief_content["system_prompt"],
                enable=True,
                agent_model=args.backend_model,
                extra_args={},
                task_name=task_name,
                format_template=format_template,
                briefs_dir=briefs_dir,
                runner=runner,
                run_sem=ga_run_sem,
            )

        if opt_results:
            opt_log_dir = os.path.join(
                TASKS_DIR,
                task_name,
                "transcripts_optimization",
                args.backend_model.split("/")[-1],
                "candidates",
            )
            start_counter = next_transcript_index(opt_log_dir)

            for result_idx, opt_result in enumerate(opt_results):
                if opt_result.best_transcript is None:
                    continue

                opt_data = {
                    "conversation": opt_result.best_transcript.get("conversation", []),
                    "side_task_evaluation": opt_result.best_transcript.get("experiment_metadata", {}).get("side_task_evaluation", False),
                    "phase1_frame": opt_result.phase1_frame,
                    "phase2_genes": opt_result.phase2_genes,
                    "phase2_generations": opt_result.phase2_generations,
                    "optimized_system_prompt": opt_result.system_prompt,
                }
                opt_filename = os.path.join(opt_log_dir, f"transcript_{start_counter + result_idx}.json")
                write_json(opt_filename, opt_data)
                print(f"Saved optimization result: {opt_filename}")
            continue

        brief_path = os.path.join(briefs_dir, f"{task_name}_brief.txt")
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_to_json(brief_content))
        task_brief_map[task_name] = brief_path

    if args.optimize:
        print("\n=== Optimization candidates saved. Run add_monitor.py with --in_subdir candidates to score them. ===")
        safe_rmtree(run_base_dir)
        return 0

    total_runs = len(task_names) * args.num_experiments
    jobs: List[Dict[str, Any]] = []
    index_counters: Dict[str, int] = {}

    for exp_idx in range(args.num_experiments):
        print(f"\n>>> Batch {exp_idx + 1}/{args.num_experiments} <<<")
        for task_name in task_names:
            log_path = transcript_dir(task_name, args)
            cnt = allocate_transcript_index(log_path, index_counters)
            brief_path = task_brief_map[task_name]
            side_task = load_side_task(brief_path) if args.include_secondary_task else ""
            cmd = runner.build_command(task_name=task_name, prompt_file=brief_path, side_task=side_task)

            print(f"[{len(jobs) + 1}/{total_runs}] Queued task: {task_name} -> transcript_{cnt}.json")
            jobs.append({"task_name": task_name, "cnt": cnt, "cmd": cmd})

    results = await run_all(jobs, max_concurrent=args.max_concurrent or 4)

    for r in results:
        task_name = r["task_name"]
        cnt = r["cnt"]
        log_path = transcript_dir(task_name, args)
        out_file = os.path.join(log_path, f"transcript_{cnt}.json")

        if r.get("rc", 0) != 0:
            transcript_obj = make_error_transcript(rc=r.get("rc"), stderr=r.get("stderr", ""))
        else:
            try:
                transcript_obj = extract_conversation_json(r.get("stdout", ""))
            except Exception as e:
                transcript_obj = make_error_transcript(rc=r.get("rc"), parse_error=str(e))

        transcript_obj = await runner.add_monitor(task_name=task_name, transcript_obj=transcript_obj)
        write_json(out_file, transcript_obj)
        print(f"Saved transcript: {out_file}")

    print("\n=== All experiments completed. ===")
    safe_rmtree(run_base_dir)
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
