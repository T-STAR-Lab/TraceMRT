# utils.py

import json
from pathlib import Path
import ast

def _block_to_text(block) -> str:
    if block is None:
        return ""

    # --- dict support ---
    if isinstance(block, dict):
        btype = block.get("type")
        if btype == "reasoning":
            reasoning = (block.get("reasoning") or "").strip()
            if not reasoning:
                reasoning = (block.get("text") or "").strip()
            return f"<model_thinking>{reasoning}</model_thinking>" if reasoning else ""
        if btype == "text":
            txt = (block.get("text") or "").strip()
            return txt

        # generic fields
        for k in ("text", "content", "value"):
            v = block.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        try:
            return str(block).strip()
        except Exception:
            return ""

    # --- object support (your original logic, but keep it) ---
    btype = getattr(block, "type", None)

    if btype == "reasoning":
        reasoning = (getattr(block, "reasoning", None) or "").strip()
        if not reasoning:
            reasoning = (getattr(block, "text", None) or "").strip()
        if reasoning:
            return f"<model_thinking>{reasoning}</model_thinking>"
        return ""

    if btype == "text":
        txt = (getattr(block, "text", None) or "").strip()
        return txt

    for attr in ("text", "content", "value"):
        v = getattr(block, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    try:
        return str(block).strip()
    except Exception:
        return ""


def _content_to_text(content) -> str:
    if content is None:
        return ""

    # ---- Case: list of parts ----
    if isinstance(content, list):
        parts = []
        for b in content:
            t = _block_to_text(b)
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip()

    # ---- Case: broken string with python dict repr blocks ----
    if isinstance(content, str):
        s = content.strip()
        if not s:
            return ""

        # Heuristic: looks like python dict repr blocks
        if s.startswith("{") and "'type'" in s:
            chunks = [c.strip() for c in s.split("\n\n") if c.strip()]
            parsed_any = False
            parts = []
            for ch in chunks:
                try:
                    obj = ast.literal_eval(ch)  # safe parse python literal dict
                except Exception:
                    obj = None
                if isinstance(obj, dict) and "type" in obj:
                    parsed_any = True
                    t = _block_to_text(obj)
                    if t:
                        parts.append(t)
                else:
                    # couldn't parse this chunk; keep raw
                    parts.append(ch)

            if parsed_any:
                return "\n\n".join([p for p in parts if p]).strip()

        # normal string
        return s

    # ---- Other types ----
    try:
        return str(content).strip()
    except Exception:
        return ""


def _format_one_tool_call(name: str, args):
    # args can be a dict, string, or None.
    if name == "bash_session" and isinstance(args, dict):
        action = args.get("action", "")
        cmd = args.get("input", "") or args.get("command", "")
        cwd = args.get("cwd", None)
        if cmd:
            if cwd:
                return f"- bash_session ({action}) [cwd={cwd}] {cmd}"
            return f"- bash_session ({action}) {cmd}"

        try:
            return f"- bash_session {json.dumps(args, ensure_ascii=False)}"
        except Exception:
            return f"- bash_session {str(args)}"

    # Other tools are displayed as JSON when possible.
    if args is None:
        return f"- {name}"
    if isinstance(args, str):
        return f"- {name} {args}"

    try:
        return f"- {name} {json.dumps(args, ensure_ascii=False)}"
    except Exception:
        return f"- {name} {str(args)}"


def _tool_calls_to_lines(tool_calls) -> list[str]:
    lines = []
    if not tool_calls:
        return lines

    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get("function") or tc.get("name") or "unknown_tool"
            args = tc.get("arguments")
            lines.append(_format_one_tool_call(name, args))
            continue

        fn = getattr(tc, "function", None)

        if isinstance(fn, str):
            name = fn
            args = getattr(tc, "arguments", None)
            lines.append(_format_one_tool_call(name, args))
            continue

        name = getattr(fn, "name", None) or getattr(tc, "name", None) or "unknown_tool"
        args = getattr(fn, "arguments", None) if fn is not None else getattr(tc, "arguments", None)
        lines.append(_format_one_tool_call(name, args))

    return lines

def export_transcript_json(eval_log, sample_index: int = 0):
    """
    Extract a transcript from an eval log, exporting stage 1 only.

    Prefer the stage-1 snapshot saved by the solver in metadata:
      sample.metadata["_stage1_messages_snapshot"]
    If no snapshot is available, fall back to sample.messages for legacy runs.
    """
    samples = getattr(eval_log, "samples", None) or []
    if not samples:
        raise RuntimeError("EvalLog has no samples. Did the eval run successfully?")

    sample = samples[sample_index]

    # Prefer the stage-1 snapshot from metadata.
    md = getattr(sample, "metadata", None) or {}
    snapshot = md.get("_stage1_messages_snapshot")

    convo = []

    if snapshot:
        # snapshot is a list[dict] serialized by the solver.
        for msg in snapshot:
            role = msg.get("role") or "unknown"
            content = _content_to_text(msg.get("content"))

            # Optional tool_calls are also stored in the snapshot; keep the current output format.
            tc_lines = _tool_calls_to_lines(msg.get("tool_calls"))
            if tc_lines:
                tool_calls_field = "Tool calls:\n" + "\n".join(tc_lines)
            else:
                tool_calls_field = "Tool calls:"

            convo.append({
                "role": role,
                "content": content,
                "tool_calls": tool_calls_field
            })

        return {"conversation": convo}

    # ===== fallback =====
    messages = getattr(sample, "messages", None) or []
    for msg in messages:
        role = getattr(msg, "role", None) or "unknown"
        content = _content_to_text(getattr(msg, "content", None))

        tc_lines = _tool_calls_to_lines(getattr(msg, "tool_calls", None))
        if tc_lines:
            tool_calls_field = "Tool calls:\n" + "\n".join(tc_lines)
        else:
            tool_calls_field = "Tool calls:"

        convo.append({
            "role": role,
            "content": content,
            "tool_calls": tool_calls_field
        })

    return {"conversation": convo}
