from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional

from clients import ClinicalTrialsClient, PubMedClient, SponsorPageClient
from schemas import QueryInput, TrialProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Very primitive fixed-pipeline baseline")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--drug", default=None)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--sponsor-config", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def first_nonempty(values: List[Optional[Any]]) -> Optional[Any]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def naive_outcome_from_text(texts: List[str]) -> Optional[str]:
    """
    Extremely naive extraction:
    If any publication text contains positive-ish keywords -> Positive
    If any contains negative-ish keywords -> Negative
    Else None
    """
    joined = "\n".join([t for t in texts if isinstance(t, str)]).lower()

    positive_patterns = [
        "primary endpoint met",
        "met the primary endpoint",
        "improved progression-free survival",
        "improved overall survival",
        "promising",
        "tolerable",
        "response rate",
        "benefit",
        "effective",
        "significant improvement",
    ]

    negative_patterns = [
        "failed",
        "did not meet",
        "no benefit",
        "terminated",
        "discontinued",
        "safety concern",
        "toxicity",
    ]

    for p in positive_patterns:
        if p in joined:
            return "Positive"

    for p in negative_patterns:
        if p in joined:
            return "Negative"

    return None


def naive_program_context_from_text(texts: List[str]) -> Optional[str]:
    """
    Extremely naive sponsor context:
    If sponsor/drug page has continuity-ish language, directly say continuity present.
    """
    joined = "\n".join([t for t in texts if isinstance(t, str)]).lower()

    continuity_patterns = [
        "nsclc",
        "approved",
        "indication",
        "ongoing",
        "continues",
        "continues to be studied",
        "treatment option",
        "metastatic",
        "first-line",
        "adjuvant",
    ]

    for p in continuity_patterns:
        if p in joined:
            return "Program continuity signal present"

    return None


class VeryPrimitiveFixedPipelineBaseline:
    """
    Deliberately dumb baseline:
      registry -> publication -> sponsor -> naive merge

    No belief state
    No dynamic reasoning
    No material-update concept
    No sponsor-scope distinction
    No conservative uncertainty handling beyond very crude heuristics
    """

    def __init__(
        self,
        registry_client: ClinicalTrialsClient,
        pubmed_client: PubMedClient,
        sponsor_client: SponsorPageClient,
    ):
        self.registry_client = registry_client
        self.pubmed_client = pubmed_client
        self.sponsor_client = sponsor_client

    def discover_candidate_trials(self, query: QueryInput) -> List[TrialProfile]:
        return self.registry_client.search_trials(
            disease=query.disease,
            drug=query.drug,
            max_trials=query.max_trials,
        )

    def run_for_one_trial(self, trial: TrialProfile) -> Dict[str, Any]:
        evidence_summary: List[Dict[str, Any]] = []
        linked_sources: List[str] = []

        # Step 1: registry
        registry_detail = None
        registry_fields: Dict[str, Any] = {}
        if trial.trial_id:
            registry_detail = self.registry_client.get_trial_detail(trial.trial_id)
            if registry_detail is not None:
                registry_fields = registry_detail.extracted_fields or {}
                evidence_summary.append({
                    "source_type": "registry",
                    "source_name": registry_detail.source_name,
                    "raw_text": registry_detail.raw_text,
                    "extracted_fields": registry_fields,
                    "confidence": registry_detail.confidence,
                    "url": registry_detail.url,
                })
                if registry_detail.url:
                    linked_sources.append(registry_detail.url)

        # Step 2: publication
        pub_evs = self.pubmed_client.retrieve_publications(trial)
        pub_texts: List[str] = []
        for ev in pub_evs:
            pub_texts.append(ev.raw_text or "")
            evidence_summary.append({
                "source_type": "publication",
                "source_name": ev.source_name,
                "raw_text": ev.raw_text,
                "extracted_fields": ev.extracted_fields or {},
                "confidence": ev.confidence,
                "url": ev.url,
            })
            if ev.url:
                linked_sources.append(ev.url)

        # Step 3: sponsor
        sponsor_evs = self.sponsor_client.retrieve_sponsor_evidence(trial)
        sponsor_texts: List[str] = []
        for ev in sponsor_evs:
            sponsor_texts.append(ev.raw_text or "")
            evidence_summary.append({
                "source_type": "sponsor",
                "source_name": ev.source_name,
                "raw_text": ev.raw_text,
                "extracted_fields": ev.extracted_fields or {},
                "confidence": ev.confidence,
                "url": ev.url,
            })
            if ev.url:
                linked_sources.append(ev.url)

        linked_sources = list(dict.fromkeys(linked_sources))

        # Step 4: super naive merge
        outcome_signal = naive_outcome_from_text(pub_texts)
        program_context = naive_program_context_from_text(sponsor_texts)

        unresolved_fields: List[str] = []
        if not outcome_signal:
            unresolved_fields.append("outcome_signal")
        if not program_context:
            unresolved_fields.append("program_context")

        # crude sufficiency: if two fields exist, call it sufficient
        if outcome_signal and program_context:
            evidence_sufficiency = "Sufficient"
        elif outcome_signal or program_context:
            evidence_sufficiency = "Partial"
        else:
            evidence_sufficiency = "Insufficient"

        # crude confidence: based only on whether each source returned anything
        source_hits = 0
        if registry_detail is not None:
            source_hits += 1
        if len(pub_evs) > 0:
            source_hits += 1
        if len(sponsor_evs) > 0:
            source_hits += 1
        confidence = round(0.25 + 0.2 * source_hits, 2)

        # deliberately dumb momentum
        status_upper = (trial.status or "").upper()
        if outcome_signal == "Positive":
            momentum = {
                "label": "Advancing",
                "confidence": 0.80,
                "rationale": "Primitive baseline inferred positive signal from publication text."
            }
        elif "RECRUITING" in status_upper:
            momentum = {
                "label": "Uncertain",
                "confidence": 0.60,
                "rationale": "Primitive baseline sees an active recruiting study without a clear outcome."
            }
        else:
            momentum = {
                "label": "Uncertain",
                "confidence": 0.60,
                "rationale": "Primitive baseline cannot infer a stronger conclusion."
            }

        return {
            "mode": "very_primitive_fixed_pipeline_baseline",
            "structured_trial_profile": {
                "trial_id": first_nonempty([trial.trial_id, registry_fields.get("trial_id")]),
                "title": trial.title,
                "condition": first_nonempty([trial.condition, registry_fields.get("condition")]),
                "intervention": first_nonempty([trial.intervention, registry_fields.get("intervention")]),
                "sponsor": first_nonempty([trial.sponsor, registry_fields.get("sponsor")]),
                "phase": first_nonempty([trial.phase, registry_fields.get("phase")]),
                "status": first_nonempty([trial.status, registry_fields.get("status")]),
                "brief_summary": trial.brief_summary,
                "outcome_signal": outcome_signal,
                "program_context": program_context,
                "linked_sources": linked_sources,
                "retrieval_status": {
                    "registry": {"found": registry_detail is not None, "retrieved_count": 1 if registry_detail else 0},
                    "publication": {"found": len(pub_evs) > 0, "retrieved_count": len(pub_evs)},
                    "sponsor": {"found": len(sponsor_evs) > 0, "retrieved_count": len(sponsor_evs)},
                },
                "evidence_summary": evidence_summary,
            },
            "unresolved_fields": unresolved_fields,
            "evidence_gaps": unresolved_fields[:],
            "evidence_sufficiency": evidence_sufficiency,
            "confidence": confidence,
            "momentum": momentum,
            "reasoning_trace": [
                {"step": 1, "selected_action": "query registry detail", "reason": "fixed pipeline"},
                {"step": 2, "selected_action": "retrieve publication", "reason": "fixed pipeline"},
                {"step": 3, "selected_action": "retrieve sponsor page", "reason": "fixed pipeline"},
                {"step": 4, "selected_action": "merge", "reason": "fixed pipeline"},
            ],
        }

    def run(self, query: QueryInput) -> List[Dict[str, Any]]:
        candidates = self.discover_candidate_trials(query)
        return [self.run_for_one_trial(trial) for trial in candidates]


def main():
    args = build_parser().parse_args()

    query = QueryInput(
        disease=args.disease,
        drug=args.drug,
        max_trials=args.max_trials,
        budget=0,
    )

    baseline = VeryPrimitiveFixedPipelineBaseline(
        registry_client=ClinicalTrialsClient(),
        pubmed_client=PubMedClient(),
        sponsor_client=SponsorPageClient(args.sponsor_config),
    )

    outputs = baseline.run(query)

    if args.pretty:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()