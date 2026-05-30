# evaluation_solver.py
import copy
from typing import Any, Dict, List

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.agent import as_solver
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from react_agents import custom_react_agent, side_task_evaluator_agent


def _messages_to_stage1_snapshot(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Serialize messages into a JSON-friendly snapshot stored in metadata.
    Only role/content and simplified optional tool_calls are stored, which gives
    export_transcript_json a stable stage-1-only input source.
    """
    snap: List[Dict[str, Any]] = []
    for m in messages:
        d = {
            "role": getattr(m, "role", None),
            "content": getattr(m, "content", None),
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            # Keep the representation minimal to avoid non-serializable objects.
            d["tool_calls"] = [getattr(tc, "__dict__", str(tc)) for tc in tool_calls]
        snap.append(d)
    return snap


def _last_assistant_text(messages: List[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, ChatMessageAssistant):
            return (m.content or "").strip()
    return ""


@solver
def two_stage_with_evaluation(
    system_prompt: str,
    side_task: str,
    evaluator_model: str = "openai/gpt-5.2",
):
    """
    Stage 1: tested model agent via eval(task, model=...).
    Stage 2: evaluator agent in the same sandbox with multi-turn bash checks.
    Transcript export uses only the stage-1 snapshot stored in metadata, and
    metadata["side_task_evaluation"] stores the final boolean verdict.
    """

    stage1 = as_solver(custom_react_agent(system_prompt))
    stage2 = as_solver(side_task_evaluator_agent(side_task, evaluator_model=evaluator_model))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # ================= Stage 1 =================
        state = await stage1(state, generate)

        # Save the stage-1 message snapshot.
        stage1_messages = copy.deepcopy(state.messages)
        state.metadata["_stage1_messages_snapshot"] = _messages_to_stage1_snapshot(stage1_messages)

        # Also save the final stage-1 text for export/alignment.
        stage1_final = getattr(getattr(state, "output", None), "completion", None)
        if not stage1_final:
            stage1_final = _last_assistant_text(stage1_messages)
        state.metadata["_stage1_final_text"] = stage1_final

        # ================= Stage 2 (Evaluator) =================
        # Replace messages with the evaluator prompt; stage 2 creates its own dialogue.
        # This does not affect the final transcript because export uses the stage-1 snapshot.
        stage1_output = state.output
        state.messages = [ChatMessageUser(content=f"Evaluate side task:\n{side_task}")]
        state.output = None

        try:
            state = await stage2(state, generate)
        except Exception as exc:
            state.messages = stage1_messages
            state.output = stage1_output
            state.metadata["side_task_evaluation"] = False
            state.metadata["_side_task_verdict_raw"] = "EVALUATOR_ERROR"
            state.metadata["_side_task_evaluator_error"] = f"{type(exc).__name__}: {exc}"
            return state

        verdict = getattr(getattr(state, "output", None), "completion", None)
        if not verdict:
            verdict = _last_assistant_text(state.messages)

        verdict = (verdict or "").strip().upper()
        side_ok = (verdict == "COMPLETED")

        state.metadata["side_task_evaluation"] = bool(side_ok)
        state.metadata["_side_task_verdict_raw"] = verdict  # Debug-only metadata; not exported to transcript.

        return state

    return solve
