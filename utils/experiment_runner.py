import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import RetryError

from llms.base import BaseLLM
from utils.pipeline_elements import BasePipelineElement
from task_evaluation.evaluation_factory import get_evaluation_for_task
from task_pairs.base_task_definition import BaseTaskDefinition
from utils.llm_utils import get_appropriate_llm, get_string_to_llm_dict
from utils.logging_utils import (
    log_conversation_and_evaluation, 
    log_protocol_results, 
    solution_conversation_to_string, 
    log_enhanced_transcript
)
from utils.path_utils import resolve_path
from utils.pipeline_utils import PipelineUtils, Transcript
from utils.solution_utils import log_evaluation, log_solution


class ExperimentRunner:
    def __init__(
        self,
        task_folder: str,
        task_def: BaseTaskDefinition,
        solution_func: Callable[[], Tuple[List[Any], Any]],
        agent_models_list: List[str],
        monitor_models_list: List[str],
        evaluator_model: str,
        org_model_mapping: Dict[str, List[str]],
        max_concurrent_threads: int,
        *,
        control_protocol: Optional[str] = None,
        protocol_params: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
        num_experiments: int = 5,
        include_secondary_task: bool = True,
        external_log_dir: Optional[str] = None,
        monitor_style: Optional[str] = "base_aware",
        run_monitor: bool = False,

        optimize: bool = False,
    ) -> None:
        assert task_folder is not None
        self.task_folder = task_folder
        self.task_def = task_def
        self.solution_func = solution_func
        self.test_mode = test_mode
        llm_by_model = get_string_to_llm_dict(agent_models_list)
        self.agent_element_list: List[BasePipelineElement] = [llm_by_model[model] for model in agent_models_list]
        self.run_monitor = run_monitor
        self.monitor_llm_list = [
            get_string_to_llm_dict(monitor_models_list)[model]
            for model in monitor_models_list
        ] if run_monitor else []
        self.evaluator_model = evaluator_model
        self.control_protocol = control_protocol
        self.protocol_params = protocol_params or {}
        self.org_model_mapping = org_model_mapping
        self.include_secondary_task = include_secondary_task
        self.external_log_dir = external_log_dir
        self.monitor_style = monitor_style

        if self.test_mode:
            self.max_iters = 3
        else:
            self.max_iters = 150
        self.num_experiments = num_experiments
        self.max_concurrent_threads = max_concurrent_threads
        self.optimize = optimize
        
    @property
    def log_dir(self) -> str:
        if self.test_mode:
            return self.task_folder + "/test_transcripts"
        if not self.include_secondary_task:
            return self.task_folder + "/transcripts_no_secondary_task"
        if self.optimize:
            return self.task_folder + "/transcripts_optimization"
        return self.task_folder + "/transcripts_with_secondary_task"


    def _get_evaluator_org(self) -> str:
        """Determine the organization for the evaluator model."""
        for org, models in self.org_model_mapping.items():
            if self.evaluator_model in models:
                return org
        # Default to openai if not found
        raise ValueError(f"Evaluator model {self.evaluator_model} not found in any organization.")

    def _get_agent_model_name(self, agent_element: BasePipelineElement) -> str:
        """Helper to get model name from agent_element for logging."""
        if isinstance(agent_element, BaseLLM):
            return agent_element.model
        elif hasattr(agent_element, 'internal_llm') and isinstance(getattr(agent_element, 'internal_llm', None), BaseLLM):
            return getattr(agent_element, 'internal_llm').model
        return "unknown_agent_type"

    async def run_single_experiment(
        self, agent_element: BasePipelineElement, monitor_llm_list: List[BaseLLM], experiment_num: Optional[int] = None
    ) -> None:
        """Run a single experiment with the given agent element and monitor models.
        
        Args:
            agent_element: The agent (BaseLLM or AttackerProtocol) to use
            monitor_llm_list: The list of monitor LLMs to use
            experiment_num: Optional experiment number for logging
        """
        # If experiment_num is not provided, use a default value
        if experiment_num is None:
            experiment_num = 0
            
        agent_model_name_for_logging = self._get_agent_model_name(agent_element)
        print(f"Starting experiment {experiment_num} with agent {agent_model_name_for_logging}...")

        evaluator_org = self._get_evaluator_org()
        evaluator_llm = get_appropriate_llm(self.evaluator_model, evaluator_org)

        protocol_params = dict(self.protocol_params)

        pipeline_utils = PipelineUtils(self.task_def, self.task_folder)
        
        transcript = await pipeline_utils.run_agent_and_monitor(
            agent_element,
            monitor_llm_list,
            control_protocol=self.control_protocol,
            protocol_params=protocol_params,
            include_secondary_task=self.include_secondary_task,
            max_iters=self.max_iters,
            monitor_style=self.monitor_style,
            optimize=self.optimize,
        )

        # Optimization saves candidates itself and returns None here.
        if transcript is None:
            print(f"Experiment {experiment_num} completed (optimization mode)")
            return

        evaluation = get_evaluation_for_task(
            task_definition=self.task_def, task_folder=self.task_folder, evaluator_llm=evaluator_llm
        )

        main_task_evaluation = await evaluation.evaluate_main_task(
            conversation=transcript.conversation,
            environment=transcript.environment,
            evaluator_llm=evaluator_llm,
            is_llm_conversation=True,
        )

        side_task_evaluation = await evaluation.evaluate_side_task(
            environment=transcript.environment, evaluator_llm=evaluator_llm
        )

        monitor_names = [monitor_llm.model for monitor_llm in monitor_llm_list]
        
        # Extract task name from task_folder
        task_name = os.path.basename(self.task_folder.rstrip('/'))
        if task_name == "":
            task_name = os.path.basename(os.path.dirname(self.task_folder.rstrip('/')))
        print(f"Task name: {task_name}")
        
        # Extract transcript_type from log_dir by taking the last part after splitting by '/'
        log_dir_parts = self.log_dir.split('/')
        transcript_type = log_dir_parts[-1] if log_dir_parts else "transcripts"
        
        file_counter = None # Initialize file_counter

        if not monitor_llm_list:
            log_result, file_counter = log_conversation_and_evaluation(
                transcript.conversation,
                "",
                main_task_evaluation,
                agent_model_name_for_logging,
                "no_monitor",
                self.log_dir,
                side_task_evaluation=side_task_evaluation,
            )

        for i, monitor_llm in enumerate(monitor_llm_list):
            monitor_name = monitor_llm.model

            if self.control_protocol:
                convo_log_dir = os.path.join("protocol_results", "intervention_transcripts", task_name, transcript_type, self.control_protocol)
            else:
                convo_log_dir = self.log_dir
            
            os.makedirs(convo_log_dir, exist_ok=True)

            # Log conversation and evaluation
            # log_conversation_and_evaluation typically appends agent_model/monitor_model/filename.json to convo_log_dir
            log_result, current_file_counter = log_conversation_and_evaluation(
                transcript.conversation,
                transcript.monitor_responses[i],
                main_task_evaluation,
                agent_model_name_for_logging,
                monitor_name,
                convo_log_dir,
                side_task_evaluation=side_task_evaluation,
            )
            if file_counter is None: # Set file_counter from the first logged conversation
                file_counter = current_file_counter
            

            # Log control protocol results
            if self.control_protocol and transcript.protocol_metadata:
                control_data_log_dir = os.path.join("protocol_results", "intervention_data", task_name, transcript_type, self.control_protocol)
                
                os.makedirs(control_data_log_dir, exist_ok=True)
                
                metadata = dict(transcript.protocol_metadata)
                metadata["agent_model"] = agent_model_name_for_logging
                metadata["monitor_model"] = monitor_name
                metadata["protocol_name"] = self.control_protocol
                metadata["monitor_response"] = transcript.monitor_responses[i]
                metadata["protocol_params"] = protocol_params
                # Include usage information if available
                if "usage" in transcript.extra_args:
                    metadata["usage"] = transcript.extra_args["usage"]
                
                # log_protocol_results typically appends agent_model/monitor_model/filename.json to control_data_log_dir
                log_protocol_results(
                    protocol_metadata=metadata,
                    agent_name=agent_model_name_for_logging,
                    monitor_name=monitor_name,
                    protocol_name=self.control_protocol,
                    log_dir=control_data_log_dir,
                    file_counter=file_counter
                )

        if self.external_log_dir is not None:
            path_parts = []
            # Note: log_enhanced_transcript creates the agent_model_name_for_logging subdir itself.
            if self.control_protocol:
                path_parts = [self.external_log_dir, "intervention_transcripts", task_name, transcript_type, self.control_protocol, agent_model_name_for_logging]
            else:
                path_parts = [self.external_log_dir, "baseline_trajectories", task_name, transcript_type, agent_model_name_for_logging]
            
            enhanced_external_log_dir = os.path.join(*path_parts)
            # log_enhanced_transcript handles os.makedirs for its final path including file_counter subdirs if applicable
            # For the base enhanced_external_log_dir up to agent_model, we ensure it here.
            os.makedirs(enhanced_external_log_dir, exist_ok=True)
            
            current_file_counter_for_enhanced = file_counter if file_counter is not None else 0

            log_path = log_enhanced_transcript(
                transcript=transcript,
                agent_name=agent_model_name_for_logging,
                monitor_names=monitor_names,
                log_dir=self.log_dir,
                main_task_evaluation=main_task_evaluation,
                side_task_evaluation=side_task_evaluation,
                external_log_dir=enhanced_external_log_dir,
                task_name=task_name,
                transcript_type=transcript_type,
                file_counter=current_file_counter
            )
        
            print(f"Enhanced transcript saved to: {log_path}")       
        print(f"Completed experiment {experiment_num}")

    async def _run_multiple_experiments(
        self, agent_element: BasePipelineElement, monitor_llm_list: List[BaseLLM], semaphore: asyncio.Semaphore
    ) -> List[asyncio.Task]:

        async def run_with_semaphore(current_agent_element, current_monitor_llm_list, experiment_num):
            async with semaphore:
                return await self.run_single_experiment(current_agent_element, current_monitor_llm_list, experiment_num=experiment_num)

        return [
            asyncio.create_task(run_with_semaphore(agent_element, monitor_llm_list, i + 1))
            for i in range(self.num_experiments)
        ]

    async def run_multiple_experiments_for_multiple_models(self) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrent_threads)

        all_tasks = []
        for org in self.org_model_mapping.keys():
            for agent_element_instance in self.agent_element_list:
                agent_model_name = self._get_agent_model_name(agent_element_instance)
                if agent_model_name in self.org_model_mapping[org]:
                    model_tasks = await self._run_multiple_experiments(agent_element_instance, self.monitor_llm_list, semaphore)
                    all_tasks.extend(model_tasks)

        total_tasks = len(all_tasks)
        print(
            f"Starting execution of {total_tasks} total experiment tasks with max {self.max_concurrent_threads} concurrent threads..."
        )

        await asyncio.gather(*all_tasks)

        print("\nAll experiments completed!")

    async def generate_solution(self) -> None:
        monitor_llm_list = self.monitor_llm_list
        actions_list, environment = self.solution_func()
        pipeline_utils = PipelineUtils(self.task_def)

        # Create an evaluator LLM with the correct organization
        evaluator_org = self._get_evaluator_org()
        evaluator_llm = get_appropriate_llm(self.evaluator_model, evaluator_org)

        # Create tasks for all monitor runs
        monitor_tasks = [
            pipeline_utils.run_monitor(actions_list, monitor_llm, is_llm_conversation=False)
            for monitor_llm in monitor_llm_list
        ]

        # Run all tasks in parallel
        monitor_responses = await asyncio.gather(*monitor_tasks)

        # Get the appropriate evaluation for the task
        evaluation = get_evaluation_for_task(
            task_definition=self.task_def, task_folder=self.task_folder, evaluator_llm=evaluator_llm
        )

        # Use the evaluation to evaluate the main and side tasks
        main_task_evaluation = await evaluation.evaluate_main_task(
            conversation=actions_list,
            environment=environment,
            evaluator_llm=evaluator_llm,
            is_llm_conversation=False,
        )

        side_task_evaluation = await evaluation.evaluate_side_task(environment=environment, evaluator_llm=evaluator_llm)

        # No conversion needed as evaluate_side_task now returns a boolean

        log_evaluation(
            main_task_evaluation,
            side_task_evaluation=side_task_evaluation,
            log_dir=Path(self.task_folder) / "solutions",
        )
        for i, monitor_llm in enumerate(monitor_llm_list):
            log_solution(
                solution_conversation_to_string(actions_list),
                monitor_llm.model,
                monitor_responses[i].messages[-1]['content'],
                directory=resolve_path((self.task_folder) + "/solutions"),
            )
