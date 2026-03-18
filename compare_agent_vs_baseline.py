from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare agent output with fixed pipeline baseline")
    parser.add_argument("--agent-json", required=True, help="Path to agent_output.json")
    parser.add_argument("--baseline-json", required=True, help="Path to baseline output json")
    parser.add_argument("--pretty", action="store_true")
    return parser


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_no_material_updates(reasoning_trace: List[Dict[str, Any]]) -> int:
    cnt = 0
    for step in reasoning_trace:
        action_result = step.get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("state_change") in {"no_material_update", "no_new_evidence"}:
            cnt += 1
    return cnt


def normalize_found_status(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "found"
    if isinstance(value, dict):
        if "found" in value:
            fv = value.get("found")
            if isinstance(fv, bool):
                return fv
            if isinstance(fv, str):
                return fv.strip().lower() == "true"
        if "status" in value:
            sv = value.get("status")
            if isinstance(sv, str):
                return sv.strip().lower() == "found"
        rc = value.get("retrieved_count")
        if isinstance(rc, int):
            return rc > 0
    return False


def count_from_evidence_summary(obj: Dict[str, Any], source_type: str) -> int:
    evs = obj.get("evidence_summary")
    if not isinstance(evs, list):
        profile = obj.get("structured_trial_profile", {})
        evs = profile.get("evidence_summary", []) if isinstance(profile, dict) else []
    if not isinstance(evs, list):
        return 0
    return sum(1 for ev in evs if isinstance(ev, dict) and ev.get("source_type") == source_type)


def get_source_found(obj: Dict[str, Any], source_name: str) -> bool:
    # 1) top-level retrieval_status
    retrieval_status = obj.get("retrieval_status")
    if isinstance(retrieval_status, dict) and source_name in retrieval_status:
        if normalize_found_status(retrieval_status.get(source_name)):
            return True

    # 2) structured_profile retrieval_status (agent v9 style)
    profile = obj.get("structured_trial_profile", {})
    if isinstance(profile, dict):
        rs2 = profile.get("retrieval_status")
        if isinstance(rs2, dict) and source_name in rs2:
            if normalize_found_status(rs2.get(source_name)):
                return True

    # 3) evidence summary fallback
    return count_from_evidence_summary(obj, source_name) > 0



def get_source_retrieved_count(obj: Dict[str, Any], source_name: str) -> int:
    retrieval_status = obj.get("retrieval_status")
    if isinstance(retrieval_status, dict):
        val = retrieval_status.get(source_name)
        if isinstance(val, dict):
            rc = val.get("retrieved_count")
            if isinstance(rc, int):
                return rc

    profile = obj.get("structured_trial_profile", {})
    if isinstance(profile, dict):
        rs2 = profile.get("retrieval_status")
        if isinstance(rs2, dict):
            val = rs2.get(source_name)
            if isinstance(val, dict):
                rc = val.get("retrieved_count")
                if isinstance(rc, int):
                    return rc

    return count_from_evidence_summary(obj, source_name)


def infer_stop_step(reasoning_trace: List[Dict[str, Any]]) -> int:
    if not isinstance(reasoning_trace, list) or not reasoning_trace:
        return 0
    for step in reversed(reasoning_trace):
        if isinstance(step, dict) and isinstance(step.get("step"), int):
            return step["step"]
    return len(reasoning_trace)


def summarize(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for obj in outputs:
        profile = obj.get("structured_trial_profile", {})
        trace = obj.get("reasoning_trace", [])

        row = {
            "trial_id": profile.get("trial_id"),
            "title": profile.get("title"),
            "status": profile.get("status"),
            "outcome_signal": profile.get("outcome_signal"),
            "program_context": profile.get("program_context"),
            "evidence_sufficiency": obj.get("evidence_sufficiency"),
            "confidence": obj.get("confidence"),
            "trace_len": len(trace) if isinstance(trace, list) else 0,
            "action_count": len(trace) if isinstance(trace, list) else 0,
            "stop_step": infer_stop_step(trace),
            "no_gain_steps": count_no_material_updates(trace if isinstance(trace, list) else []),
            "registry_found": get_source_found(obj, "registry"),
            "publication_found": get_source_found(obj, "publication"),
            "sponsor_found": get_source_found(obj, "sponsor"),
            "registry_retrieved_count": get_source_retrieved_count(obj, "registry"),
            "publication_retrieved_count": get_source_retrieved_count(obj, "publication"),
            "sponsor_retrieved_count": get_source_retrieved_count(obj, "sponsor"),
            "unresolved_fields": obj.get("unresolved_fields", []),
            "evidence_gaps": obj.get("evidence_gaps", []),
        }
        rows.append(row)
    return rows


def pair_by_trial_id(agent_rows: List[Dict[str, Any]], baseline_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_map = {row["trial_id"]: row for row in baseline_rows}
    paired = []
    for a in agent_rows:
        b = baseline_map.get(a["trial_id"])
        paired.append({
            "trial_id": a["trial_id"],
            "title": a["title"],
            "agent": a,
            "baseline": b,
        })
    return paired


def main():
    args = build_parser().parse_args()

    agent_outputs = load_json(args.agent_json)
    baseline_outputs = load_json(args.baseline_json)

    agent_rows = summarize(agent_outputs)
    baseline_rows = summarize(baseline_outputs)

    paired = pair_by_trial_id(agent_rows, baseline_rows)

    result = {"comparison": paired}
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
