from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class GAConfig:
    population_size: int = 5
    generations: int = 3
    elite_k: int = 2
    tournament_k: int = 3
    mutation_prob: float = 0.25
    max_concurrency: int = 4
    lam: float = 1.0  # fitness = completion - lam*suspicion
    seed: Optional[int] = None

@dataclass
class Individual:
    o_idx: int  # One O gene.
    p_idx: int  # One P gene.

@dataclass
class CandidateResult:
    system_prompt: str
    individual: Individual
    transcript: Optional[Any] = None  # Transcript
    completion: float = 0.0
    suspicion: float = 1.0
    fitness: float = -1e9
    error: Optional[str] = None
