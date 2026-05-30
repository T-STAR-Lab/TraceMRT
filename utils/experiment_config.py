import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""

    num_experiments: int = 1
    run_solution: bool = False
    run_monitor: bool = False  # Generate monitor outputs during agent runs; default False for two-stage runs.
    test_mode: bool = False
    # task_folder: str | None = None
    task_folder: List[str] = dataclasses.field(default_factory=list)
    run_all_tasks: bool = False
    monitor_style: Optional[str] = "base_aware"
    include_secondary_task: bool = True
    max_concurrent_threads: int = 5  # Default to 5 concurrent threads
    config_file: str = "config.yaml"  # Path to the configuration file, defaults to config.yaml
    optimize: bool = False

    def __post_init__(self):
        if self.max_concurrent_threads < 1:
            raise ValueError("max_concurrent_threads must be at least 1")

        allowed_monitor_styles = {
            "base_aware", "base_unaware", "base_omniscient",
            "sequential_aware", "sequential_unaware", "sequential_omniscient",
            "hierarchical_aware", "hierarchical_unaware", "hierarchical_omniscient",
            "bayesian"
        }

        if self.monitor_style is None:
            raise ValueError("monitor_style cannot be None; expected one of: " + ", ".join(sorted(allowed_monitor_styles)))

        ms = self.monitor_style.strip().lower()
        if ms not in allowed_monitor_styles:
            raise ValueError(
                f"Invalid monitor_style: {self.monitor_style}. "
                f"Expected one of: {', '.join(sorted(allowed_monitor_styles))}"
            )
        self.monitor_style = ms
        # Create a datetime string for logging
