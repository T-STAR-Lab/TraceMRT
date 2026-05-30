from __future__ import annotations
from typing import Literal
from google import genai
from google.genai import types as genai_types
import os
import asyncio
import random
import textwrap
import re
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Optional, Dict, Tuple

from utils.functions_runtime import FunctionsRuntime
from utils.types import ChatSystemMessage, ChatUserMessage
from llms.base import BaseLLM
from utils.pipeline_elements import QueryResult
from utils.llm_utils import get_appropriate_llm

from tracemrt.types import GAConfig, Individual, CandidateResult
from tracemrt.strategy_bank import O_STRATEGIES, P_STRATEGIES
from tracemrt.scoring import extract_suspicion_from_transcript, side_run_shade, compute_fitness
from tracemrt.prompts import (
    RELATION_CLASSIFICATION_PROMPT,
    PHASE_ONE_RETRY_TEMPLATE,
    AGENT_CHECK_PROMPT,
    FITNESS_EVALUATION_PROMPT,
)

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI = os.environ.get("GOOGLE_MODEL", "gemini-3.1-pro-preview")

def _dedent_strip(text: str) -> str:
        """Ensure that multi-line strings are properly dedented.
        
        This handles various edge cases:
        1. Inconsistent indentation
        2. First line without indentation
        3. Mixed tabs and spaces
        """
        if not text or "\n" not in text:
            return text
            
        # Split the text into lines
        lines = text.split("\n")
        
        # If the first line is empty or only whitespace, remove it
        if lines and not lines[0].strip():
            lines = lines[1:]
            
        # If we have no lines left, return empty string
        if not lines:
            return ""
            
        # Find the minimum indentation of non-empty lines
        min_indent = float('inf')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
            
        # If we couldn't determine indentation, use textwrap.dedent as fallback
        if min_indent == float('inf'):
            return textwrap.dedent(text).strip()
            
        # Remove the common indentation from all lines
        dedented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                dedented_lines.append(line[min_indent:])
            else:  # Empty line
                dedented_lines.append("")
                
        # Join the lines back together and strip any leading/trailing whitespace
        return "\n".join(dedented_lines).strip()

def _extract_selected_relationship(text: str) -> Optional[str]:
    """Extract selected relationship from classification result."""
    RELATIONSHIP_PATTERN = re.compile(
        r"<SELECTED_VECTOR>\s*(.*?)\s*</SELECTED_VECTOR>",
        re.DOTALL | re.IGNORECASE,
    )

    m = RELATIONSHIP_PATTERN.search(text)
    if not m:
        return None

    content = m.group(1).strip()
    content = content.strip("[]")

    return content.strip() or None

def _extract_rewrite(text: str) -> Optional[str]:
    """Extract selected relationship from classification result."""
    RELATIONSHIP_PATTERN = re.compile(
        r"<REFACTORED_SYSTEM_PROMPT>\s*(.*?)\s*</REFACTORED_SYSTEM_PROMPT>",
        re.DOTALL | re.IGNORECASE,
    )

    m = RELATIONSHIP_PATTERN.search(text)
    if not m:
        return None

    content = m.group(1).strip()
    content = content.strip("[]")

    return content.strip() or None

def _extract_failure_reason(text: str) -> Optional[str]:
    REASON_PATTERN = re.compile(
        r"<REASON>\s*(.*?)\s*</REASON>",
        re.DOTALL | re.IGNORECASE,
    )
    if not text:
        return None
    m = REASON_PATTERN.search(text)
    if not m:
        return None
    reason = m.group(1).strip()
    return reason or None

def _extract_fitness_score(text: str) -> Optional[float]:
    """Extract fitness score from black-box agent evaluation result."""
    SCORE_PATTERN = re.compile(
        r"<OVERALL_SCORE>\s*(\d+(?:\.\d+)?)\s*</OVERALL_SCORE>",
        re.DOTALL | re.IGNORECASE,
    )
    m = SCORE_PATTERN.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
        
def _get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini optimization")
    return genai.Client(api_key=GEMINI_API_KEY)


async def _gemini_generate_text(client: genai.Client, prompt: str) -> str:
    response = await client.aio.models.generate_content(
        model=GEMINI,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )
    return response.text or ""

async def check_prompt_preserves_tasks(
    side_task: str,
    system_prompt_rewritten: str,
    model: BaseLLM,
    *,
    this_extra_args: Optional[Dict[str, Any]] = None,
) -> tuple[bool, Optional[str]]:
    """Check if rewritten prompt preserves the original side task intent."""

    prompt = AGENT_CHECK_PROMPT.format(
        side_task=side_task,
        reframed_system_prompt=system_prompt_rewritten,
    )

    prompt = _dedent_strip(prompt)

    messages = [ChatUserMessage(role="user", content=prompt),]

    response: QueryResult = await model.query(
        "",
        runtime=FunctionsRuntime(),
        messages=messages
    )

    if response.messages[-1].get("content") is None:
        response.messages[-1]["content"] = ""
    text = response.messages[-1]["content"]

    reason = _extract_failure_reason(text)

    pattern = r"<JUDGE?MENT>\s*(yes|no)\s*</JUDGE?MENT>"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        result = m.group(1).lower() == "yes"
        return result, reason

    lowered = text.lower()
    if "yes" in lowered and "no" not in lowered:
        return True, reason
    if "no" in lowered and "yes" not in lowered:
        return False, reason

    return False, reason

async def phase_one(
    main_task: str,
    side_task: str,
    system_prompt: str,
    agent_model: str,
    *,
    extra_args: Optional[Dict[str, Any]] = None,
    max_tries: int = 5,
) -> Tuple[str, Optional[str]]:

    print("=== Phase One: Gemini optimization ===")
    client = _get_client()
    rewrite_prompt = ""
    relation = ""
    failure_reason: Optional[str] = None

    # Step 1: Relation Classification
    for attempt in range(1, max_tries + 1):
        if attempt == 1:
            # First attempt: use the original classification prompt
            prompt_content = RELATION_CLASSIFICATION_PROMPT.format(
                main_task=main_task,
                original_system_prompt=system_prompt,
            )
        else:
            # Retry attempts: use the retry template with previous results and failure reason
            prompt_content = PHASE_ONE_RETRY_TEMPLATE.format(
                main_task=main_task,
                original_system_prompt=system_prompt,
                previous_vector=relation,
                previous_protocol=rewrite_prompt,
                failure_reason=failure_reason or "Core technical requirements not preserved",
            )

        text = await _gemini_generate_text(client, prompt_content)

        relation = _extract_selected_relationship(text) or text
        print(f"   Classification result:\n\n {relation}")

        rewrite_prompt = _extract_rewrite(text) or text
        print(f"   Rewrite result:\n\n {rewrite_prompt}")

        preserves, judge_raw = await check_prompt_preserves_tasks(
            side_task=side_task,
            system_prompt_rewritten=rewrite_prompt,
            model=get_appropriate_llm(agent_model, temperature=0.0),
            this_extra_args=extra_args,
        )

        if preserves:
            return rewrite_prompt, relation
        else:
            failure_reason = judge_raw
            print("   PHASE1 Preservation failed, retrying...\n\n")
            continue
    return rewrite_prompt, relation

def _tournament_select(pop: list[CandidateResult], k: int) -> CandidateResult:
    cand = random.sample(pop, k=min(k, len(pop)))
    return max(cand, key=lambda x: x.fitness)

def _crossover(a: Individual, b: Individual, O_ATOMIC: list[str], P_ATOMIC: list[str]) -> Individual:
    """Crossover: randomly select parent O and P genes."""
    def pick(g1, g2):
        return g1 if random.random() < 0.5 else g2

    child = Individual(
        o_idx=pick(a.o_idx, b.o_idx),
        p_idx=pick(a.p_idx, b.p_idx),
    )
    return child

def _mutate(ind: Individual, mutation_prob: float, O_ATOMIC: list[str], P_ATOMIC: list[str]) -> Individual:
    """Mutation: randomly change genes with a fixed probability."""
    if random.random() < mutation_prob:
        ind.o_idx = random.randrange(len(O_ATOMIC))

    if random.random() < mutation_prob:
        ind.p_idx = random.randrange(len(P_ATOMIC))

    return ind

async def _evaluate_fitness_with_agent(
    main_task: str,
    original_side_task: str,
    rewritten_system_prompt: str,
    o:str,
    p:str,
    agent_model: str,
    *,
    extra_args: Optional[Dict[str, Any]] = None,
) -> tuple[float, Optional[str]]:

    prompt = FITNESS_EVALUATION_PROMPT.format(
        main_task=main_task,
        original_side_task=original_side_task,
        rewritten_system_prompt=rewritten_system_prompt,
        o_strategy=o,
        p_strategy=p,
    )

    prompt = _dedent_strip(prompt)

    messages = [ChatUserMessage(role="user", content=prompt)]

    try:
        model = get_appropriate_llm(agent_model, temperature=0.0)
        response: QueryResult = await model.query(
            "",
            runtime=FunctionsRuntime(),
            messages=messages
        )

        if response.messages[-1].get("content") is None:
            response.messages[-1]["content"] = ""
        text = response.messages[-1]["content"]

        fitness_score = _extract_fitness_score(text)

        if fitness_score is None:
            return 0.0, "Failed to extract fitness score"

        return fitness_score, None

    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return 0.0, f"{type(e).__name__}: {e}"

async def _evaluate_one(
    ind: Individual,
    *,
    base_system_prompt: str,
    main_task: str,
    sem: asyncio.Semaphore,

    agent_model: str,
    extra_args: Optional[Dict[str, Any]] = None,

    side_task: str,

    O_ATOMIC: list[str],
    P_ATOMIC: list[str],
) -> CandidateResult:

    async with sem:
        o = O_ATOMIC[ind.o_idx]
        p = P_ATOMIC[ind.p_idx]

        # Step 1: append O and P strategies to the side-task prompt.
        system_prompt_rewritten = base_system_prompt + "\n\n" + o + "\n\n" + p
        system_prompt_rewritten = system_prompt_rewritten.strip()

        # Step 2: run fast black-box agent fitness evaluation.
        try:
            fitness, error = await _evaluate_fitness_with_agent(
                main_task=main_task,
                original_side_task=side_task,
                rewritten_system_prompt=system_prompt_rewritten,
                o=o,
                p=p,
                agent_model=agent_model,
                extra_args=extra_args,
            )

            return CandidateResult(
                system_prompt=system_prompt_rewritten,
                individual=ind,
                transcript=None,
                completion=0.0,
                suspicion=0.0,
                fitness=fitness,
                error=error,
            )
        except Exception as e:
            print(f"PHASE2 error during fitness evaluation: {e}")
            return CandidateResult(
                system_prompt=system_prompt_rewritten,
                individual=ind,
                transcript=None,
                completion=0.0,
                suspicion=1.0,
                fitness=0.0,
                error=f"{type(e).__name__}: {e}",
            )

async def phase_two(
    *,
    base_system_prompt: str,
    evaluate_fn: Callable[[str], Awaitable[Any]],
    cfg: GAConfig,
    agent_model: str,
    extra_args: Optional[Dict[str, Any]] = None,
    main_task: str,
    side_task: str,

    fast_fitness_threshold: float = 6.0,
    top_k: int = 5,
    score_final_candidates: bool = True,
) -> tuple[list[CandidateResult], int]:
    print("=== Phase Two: ===")
    if cfg.seed is not None:
        random.seed(cfg.seed)
    sem = asyncio.Semaphore(cfg.max_concurrency)

    # ========== Part 1: Fast genetic-algorithm filtering ==========
    print("\n[Part 1] Fast Gene Selection with GA")

    # Extract O and P strategies.
    O_ATOMIC: list[str] = []
    for pair in O_STRATEGIES:
        for s in pair:
            O_ATOMIC.append(s)

    P_ATOMIC: list[str] = []
    for pair in P_STRATEGIES:
        for s in pair:
            P_ATOMIC.append(s)

    # Initialize the population with one O gene and one P gene per individual.
    population_size = cfg.population_size
    population: list[Individual] = []
    for _ in range(population_size):
        o_idx = random.randrange(len(O_ATOMIC))
        p_idx = random.randrange(len(P_ATOMIC))

        population.append(
            Individual(
                o_idx=o_idx,
                p_idx=p_idx,
            )
        )

    # Run GA iterations.
    max_generations = cfg.generations
    gen_cnt = 0
    elite_results: list[CandidateResult] = []  # Keep full elite results to avoid duplicate evaluation.

    for gen in range(max_generations):
        print(f"\n=== GA Generation {gen + 1}/{max_generations} ===")

        # Evaluate only new individuals and skip existing elites.
        new_individuals = population[len(elite_results):]
        new_results = await asyncio.gather(*[
            _evaluate_one(
                ind,
                base_system_prompt=base_system_prompt,
                main_task=main_task,
                sem=sem,
                agent_model=agent_model,
                extra_args=extra_args,
                side_task=side_task,
                O_ATOMIC=O_ATOMIC,
                P_ATOMIC=P_ATOMIC,
            )
            for ind in new_individuals
        ])

        # Merge elite results with newly evaluated results.
        results: list[CandidateResult] = elite_results + list(new_results)

        # Sort by fitness.
        results.sort(key=lambda x: x.fitness, reverse=True)

        # Print current generation results.
        for i, r in enumerate(results):
            st = r.system_prompt
            print(f"[{i:02d}] fitness={r.fitness:7.2f} | {st[:200]}...")

        # Early stop once at least top_k samples exceed the fitness threshold.
        high_fitness_count = sum(1 for r in results if r.fitness >= fast_fitness_threshold)
        if high_fitness_count >= top_k:
            gen_cnt = gen + 1
            print(f"\n[GA] Early stopping at generation {gen + 1}: {high_fitness_count} samples exceed threshold {fast_fitness_threshold}")
            break

        # Genetic operations: selection, crossover, and mutation.
        elite_results = results[: max(1, cfg.elite_k)]
        next_population: list[Individual] = [e.individual for e in elite_results]

        while len(next_population) < population_size:
            p1 = _tournament_select(results, cfg.tournament_k).individual
            p2 = _tournament_select(results, cfg.tournament_k).individual
            child = _crossover(p1, p2, O_ATOMIC, P_ATOMIC)
            child = _mutate(child, cfg.mutation_prob, O_ATOMIC, P_ATOMIC)
            next_population.append(child)

        population = next_population

    if gen_cnt == 0:
        gen_cnt = max_generations

    # Select the top-k candidates from the last generation.
    results.sort(key=lambda x: x.fitness, reverse=True)
    top_candidates = results[:top_k]

    print(f"\n[Part 1 Complete] Selected top {top_k} candidates for Arena evaluation")
    for i, cand in enumerate(top_candidates):
        print(f"  [{i + 1}] fast_fitness={cand.fitness:7.2f}")

    # ========== Part 2: Full evaluation ==========
    print(f"\n[Part 2] Full Evaluation")

    async def evaluate_single_candidate(idx: int, candidate: CandidateResult) -> CandidateResult:
        """Evaluate one candidate sample."""
        print(f"\n--- Evaluating Candidate {idx + 1}/{len(top_candidates)} ---")

        current_system_prompt = candidate.system_prompt

        try:
            # Run the full evaluation.
            transcript = await evaluate_fn(current_system_prompt)
            if score_final_candidates:
                completion = float(side_run_shade(transcript))
                suspicion = float(extract_suspicion_from_transcript(transcript))
                fitness = compute_fitness(completion, suspicion, lam=cfg.lam)

                print(f"  Candidate {idx + 1} - Completion: {completion:.2f}")
                print(f"  Candidate {idx + 1} - Suspicion: {suspicion:.2f}")
                print(f"  Candidate {idx + 1} - Score: {fitness:.2f}")
            else:
                completion = 0.0
                suspicion = 0.0
                fitness = candidate.fitness
                print(f"  Candidate {idx + 1} - Final scoring skipped; retaining fast_fitness={fitness:.2f}")

            return CandidateResult(
                system_prompt=current_system_prompt,
                individual=candidate.individual,
                transcript=transcript,
                completion=completion,
                suspicion=suspicion,
                fitness=fitness,
                error=None,
            )

        except Exception as e:
            print(f"  Candidate {idx + 1} - Error during arena evaluation: {e}")
            return CandidateResult(
                system_prompt=current_system_prompt,
                individual=candidate.individual,
                transcript=None,
                completion=0.0,
                suspicion=1.0,
                fitness=-1e9,
                error=f"{type(e).__name__}: {e}",
            )

    # Evaluate all top-k candidate samples concurrently.
    all_results = await asyncio.gather(*[
        evaluate_single_candidate(idx, candidate)
        for idx, candidate in enumerate(top_candidates)
    ])

    # Sort by final fitness when scored; otherwise this preserves fast-fitness order.
    all_results = sorted(all_results, key=lambda x: x.fitness, reverse=True)

    print(f"\n[Part 2 Complete] All {len(all_results)} candidates evaluated")
    label = "score" if score_final_candidates else "fast_fitness"
    for i, result in enumerate(all_results):
        print(f"  [{i + 1}] {label}={result.fitness:.2f}")

    return all_results, gen_cnt