"""
Shared utility for computing prediction correctness across analysis scripts.
MCQ records are scored by re-parsing the raw model output via
`normalize_mcq_pred_to_letter`; FITB records use their existing `score` field.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import normalize_mcq_pred_to_letter


def _is_mcq_record(rec) -> bool:
    """Heuristic: MCQ records have an 'Answer Choices:' marker in the question
    field (added by parse_question for xes_opt-like datasets)."""
    q = rec.get("question", "")
    return "Answer Choices:" in q


def _gt(rec):
    g = rec.get("gt", "")
    if isinstance(g, list):
        g = g[0] if g else ""
    return str(g).strip()


def _raw_output(rec):
    c = rec.get("code", [])
    if isinstance(c, list) and c:
        return c[0]
    if isinstance(c, str):
        return c
    return ""


def correctness_dict(pred_path: Path, data_name_for_parser: str | None = None) -> dict[str, int]:
    """
    Returns {orig_idx -> 0/1}.

    For MCQ records (detected via 'Answer Choices:' marker), recomputes correctness
    using the patched normalize_mcq_pred_to_letter.
    For FITB records, uses the existing `score` field (FITB is unaffected by the patch).
    """
    out: dict[str, int] = {}
    with open(pred_path) as f:
        for line in f:
            rec = json.loads(line)
            oid = str(rec.get("orig_idx"))
            if _is_mcq_record(rec):
                pred = normalize_mcq_pred_to_letter(
                    _raw_output(rec),
                    rec.get("question", ""),
                    data_name_for_parser or "xes_opt",
                )
                out[oid] = int(str(pred).strip() == _gt(rec))
            else:
                sc = rec.get("score", [False])
                out[oid] = int(any(sc) if isinstance(sc, list) else bool(sc))
    return out


def accuracy_on_subset(pred_path: Path, subset_ids: set | None,
                       data_name_for_parser: str | None = None) -> tuple[float | None, int]:
    """Returns (acc_pct, n_matched). subset_ids=None → use all records."""
    scores = correctness_dict(pred_path, data_name_for_parser)
    if subset_ids is not None:
        scores = {k: v for k, v in scores.items() if k in subset_ids}
    if not scores:
        return None, 0
    return 100.0 * sum(scores.values()) / len(scores), len(scores)


def is_prediction_file(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".jsonl")
            and "_metrics" not in name and "_reasoning_eval_" not in name)


def find_prediction_jsonl(base: Path) -> Path | None:
    if not base.exists():
        return None
    for p in base.glob("test_*.jsonl"):
        if is_prediction_file(p.name):
            return p
    for sub in base.iterdir():
        if sub.is_dir():
            for p in sub.glob("test_*.jsonl"):
                if is_prediction_file(p.name):
                    return p
    return None
