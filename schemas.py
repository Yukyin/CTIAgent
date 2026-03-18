from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    REGISTRY = "registry"
    PUBLICATION = "publication"
    SPONSOR = "sponsor"


class EvidenceSufficiency(str, Enum):
    INSUFFICIENT = "Insufficient"
    PARTIAL = "Partial"
    SUFFICIENT = "Sufficient"


class MomentumLabel(str, Enum):
    ADVANCING = "Advancing"
    UNCERTAIN = "Uncertain"
    STALLED = "Stalled"


class ActionType(str, Enum):
    QUERY_REGISTRY_DETAIL = "query registry detail"
    RETRIEVE_PUBLICATION = "retrieve publication"
    RETRIEVE_SPONSOR_PAGE = "retrieve sponsor page"
    PERFORM_ALIAS_DISAMBIGUATION = "perform alias disambiguation"
    STOP = "stop"


class QueryInput(BaseModel):
    disease: str
    drug: Optional[str] = None
    max_trials: int = 5
    budget: int = 5


class EvidenceItem(BaseModel):
    source_type: SourceType
    source_name: str
    raw_text: str
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    url: Optional[str] = None
    llm_semantic_signals: Optional[Dict[str, Any]] = None


class TrialProfile(BaseModel):
    trial_id: Optional[str] = None
    title: Optional[str] = None
    condition: Optional[str] = None
    intervention: Optional[str] = None
    sponsor: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    brief_summary: Optional[str] = None
    outcome_signal: Optional[str] = None
    program_context: Optional[str] = None
    linked_sources: List[str] = Field(default_factory=list)


class BeliefState(BaseModel):
    trial_profile: TrialProfile
    known_fields: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    evidence_sufficiency: EvidenceSufficiency = EvidenceSufficiency.INSUFFICIENT
    confidence: float = 0.2
    step_count: int = 0
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    sufficiency_rationale: Optional[str] = None
    evidence_gaps: List[str] = Field(default_factory=list)
    retrieval_status: Dict[str, str] = Field(default_factory=lambda: {
        "registry": "not_attempted",
        "publication": "not_attempted",
        "sponsor": "not_attempted",
    })
    sponsor_context_scope: Optional[str] = None


class TrialIntelligenceOutput(BaseModel):
    structured_trial_profile: Dict[str, Any]
    unresolved_fields: List[str]
    evidence_gaps: List[str] = Field(default_factory=list)
    ambiguities: List[str]
    evidence_sufficiency: str
    confidence: float
    momentum: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
