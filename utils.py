from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip()


def safe_lower(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def field_is_missing(value: Optional[str]) -> bool:
    return value is None or str(value).strip() == ""


def looks_like_broad_solid_tumor_trial(text: str) -> bool:
    t = safe_lower(text)
    broad_markers = [
        "advanced solid tumor",
        "advanced solid tumours",
        "solid tumor",
        "solid tumours",
        "neoplasms",
        "advanced cancer",
        "metastatic cancer",
    ]
    return any(m in t for m in broad_markers)


def extract_simple_outcome_signal(text: str) -> Tuple[Optional[str], float]:
    t = safe_lower(text)
    positive_patterns = [
        "primary endpoint met",
        "met the primary endpoint",
        "improved progression-free survival",
        "improved overall survival",
        "overall survival benefit",
        "progression-free survival benefit",
        "significant benefit",
        "positive trial",
        "statistically significant",
        "objective response rate improved",
        "durable response",
    ]
    negative_patterns = [
        "failed to meet",
        "did not meet",
        "negative trial",
        "no significant difference",
        "safety concern",
        "futility",
        "inferior to",
        "worse than",
    ]
    mixed_patterns = [
        "mixed results",
        "benefit in subgroup",
        "numerical improvement",
        "exploratory benefit",
        "trend toward improvement",
    ]
    for p in positive_patterns:
        if p in t:
            return "Positive", 0.75
    for p in negative_patterns:
        if p in t:
            return "Negative", 0.75
    for p in mixed_patterns:
        if p in t:
            return "Mixed", 0.60
    return None, 0.3


def extract_simple_program_context(text: str) -> Tuple[Optional[str], float]:
    t = safe_lower(text)
    continuity_patterns = [
        "listed indication",
        "continues to be studied",
        "ongoing development",
        "approved in nsclc",
        "remains a focus area",
        "continues in phase",
        "approved for",
        "indicated for",
        "remains available",
        "currently recruiting",
        "ongoing phase 3",
    ]
    discontinuity_patterns = [
        "program discontinued",
        "development halted",
        "withdrawn",
        "stopped development",
        "discontinued in",
    ]
    for p in continuity_patterns:
        if p in t:
            return "Program continuity signal present", 0.7
    for p in discontinuity_patterns:
        if p in t:
            return "Program discontinuity signal present", 0.7
    return None, 0.3


def compact_state_summary(state: Any) -> Dict[str, Any]:
    return {
        "trial_profile": state.trial_profile.model_dump(),
        "known_fields": list(state.known_fields.keys()),
        "missing_fields": state.missing_fields,
        "evidence_gaps": getattr(state, "evidence_gaps", []),
        "retrieval_status": getattr(state, "retrieval_status", {}),
        "sponsor_context_scope": getattr(state, "sponsor_context_scope", None),
        "ambiguities": state.ambiguities,
        "evidence_sufficiency": getattr(state.evidence_sufficiency, "value", state.evidence_sufficiency),
        "confidence": state.confidence,
        "evidence_inventory": [
            {
                "source_type": getattr(ev.source_type, "value", ev.source_type),
                "source_name": ev.source_name,
                "confidence": ev.confidence,
                "url": ev.url,
                "extracted_fields": ev.extracted_fields,
                "llm_semantic_signals": ev.llm_semantic_signals,
            }
            for ev in state.evidence_items
        ],
    }
