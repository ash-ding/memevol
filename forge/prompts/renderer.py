"""Prompt rendering: turn a (version, sanity_enabled, active_datasets)
tuple into final system / task / fix prompt strings.

All the version-specific text lives in `forge/prompts/templates/<stem>.py` —
this module only contains the version-agnostic helper logic for sentinel
substitution, dataset filtering, and fix-trace truncation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .loader import load_template_module


def _normalize_active_datasets(
    active_datasets: Optional[Iterable[str]],
    dataset_info: dict,
    dataset_render_order: List[str],
) -> List[str]:
    """Map dataset registration names → canonical shape group keys, in render order.

    longmemeval_s + longmemeval_m share the same shape doc → both → "longmemeval".
    Unknown names are silently dropped (defensive against typos / future names).
    None / empty → all groups in canonical order (back-compat: render full prompt).
    """
    if not active_datasets:
        return list(dataset_render_order)
    seen = set()
    for ds in active_datasets:
        key = "longmemeval" if ds.startswith("longmemeval") else ds
        if key in dataset_info:
            seen.add(key)
    return [k for k in dataset_render_order if k in seen]


def _build_dataset_subs(active: List[str], dataset_info: dict) -> dict:
    """Compute the 6 dataset-conditional sentinel substitutions."""
    n = len(active)
    names_csv = ", ".join(dataset_info[k]["display_name"] for k in active)

    if n == 0:
        # Defensive: shouldn't happen since _normalize falls back to all three,
        # but if it does, leave intro vague.
        eval_intro = (
            "Your memory system is evaluated on long-context QA benchmarks. "
            "It must handle the `recorder.init` shape provided at runtime."
        )
    elif n == 1:
        eval_intro = (
            f"Your memory system is evaluated on the {names_csv} benchmark. "
            f"See the dataset shape and dispatch pattern below."
        )
    else:
        eval_intro = (
            f"Your memory system is evaluated on MULTIPLE benchmarks "
            f"(currently {names_csv}). Each benchmark exposes a different "
            f"`recorder.init` shape, so it must handle all of them — either "
            f"with dataset-agnostic logic, or by dispatching on the keys of "
            f"`recorder.init`."
        )

    qa_meta = "\n".join(dataset_info[k]["qa_metadata"] for k in active)
    relevant = "\n".join(dataset_info[k]["relevant"] for k in active)
    shapes = "\n".join(dataset_info[k]["shape"] for k in active)

    update_lines = []
    retrieve_lines = []
    for i, k in enumerate(active):
        kw = "if" if i == 0 else "elif"
        info = dataset_info[k]
        update_lines.append(
            f"    {kw} {info['dispatch_check']}:\n"
            f"        {info['dispatch_update_comment']}\n"
            f"        ..."
        )
        retrieve_lines.append(
            f"    {kw} {info['dispatch_check']}:\n"
            f"        {info['dispatch_retrieve_comment']}\n"
            f"        ..."
        )

    return {
        "<<EVAL_INTRO_BLOCK>>": eval_intro,
        "<<TRACE_QA_METADATA_BLOCK>>": qa_meta,
        "<<TRACE_RELEVANT_BLOCK>>": relevant,
        "<<DATASET_SHAPES_BLOCK>>": shapes,
        "<<DISPATCH_UPDATE_BRANCHES>>": "\n".join(update_lines),
        "<<DISPATCH_RETRIEVE_BRANCHES>>": "\n".join(retrieve_lines),
    }


def build_proposer_system(
    *,
    sanity_enabled: bool = True,
    active_datasets: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
) -> str:
    """Render the proposer SYSTEM prompt for a given prompt-template version.

    Args:
      sanity_enabled  — False when this run will NOT run the sanity layer
                        (cfg.sanity.enabled=false, or cfg.status=devtest).
      active_datasets — iterable of dataset registration names from
                        cfg["datasets"] keys (e.g. ["dynamicmem", "locomo",
                        "longmemeval_s"]). None / empty → render all known
                        shape groups (back-compat).
      version         — prompt template version stem. None / "latest" /
                        "" → resolve via templates/_default.
    """
    mod = load_template_module(version)

    active = _normalize_active_datasets(
        active_datasets, mod.DATASET_INFO, mod.DATASET_RENDER_ORDER
    )
    subs = {
        **(mod.SANITY_ON_SUBS if sanity_enabled else mod.SANITY_OFF_SUBS),
        **_build_dataset_subs(active, mod.DATASET_INFO),
    }
    out = mod.SYSTEM_TEMPLATE
    for sentinel, replacement in subs.items():
        out = out.replace(sentinel, replacement)
    return out


def proposer_task_prompt(
    new_dir_rel: str,
    *,
    version: Optional[str] = None,
) -> str:
    """Per-iteration TASK prompt. All paths are workspace-relative."""
    mod = load_template_module(version)
    return mod.TASK_PROMPT_TEMPLATE.format(new_dir_rel=new_dir_rel)


def proposer_fix_prompt(
    new_dir_rel: str,
    error_trace: str,
    *,
    version: Optional[str] = None,
) -> str:
    """Prompt issued after a sanity-check failure: Read → diagnose → Edit."""
    lines = error_trace.splitlines()
    if len(lines) > 90:
        error_trace = (
            "\n".join(lines[:60])
            + "\n... [truncated] ...\n"
            + "\n".join(lines[-30:])
        )
    mod = load_template_module(version)
    return mod.FIX_PROMPT_TEMPLATE.format(
        new_dir_rel=new_dir_rel, error_trace=error_trace
    )
