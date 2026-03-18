from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
from schemas import (
    ActionType,
    BeliefState,
    EvidenceItem,
    EvidenceSufficiency,
    MomentumLabel,
    QueryInput,
    TrialIntelligenceOutput,
    TrialProfile,
)
from utils import (
    clip,
    compact_state_summary,
    extract_simple_outcome_signal,
    extract_simple_program_context,
    field_is_missing,
    safe_lower,
)


class TrialIntelligenceAgent:
    def __init__(self, registry_client, pubmed_client, sponsor_client, llm_reasoner=None):
        self.registry_client = registry_client
        self.pubmed_client = pubmed_client
        self.sponsor_client = sponsor_client
        self.llm_reasoner = llm_reasoner

    def discover_candidate_trials(self, query: QueryInput) -> List[TrialProfile]:
        return self.registry_client.search_trials(query.disease, query.drug, query.max_trials)

    def initialize_belief_state(self, trial: TrialProfile) -> BeliefState:
        known = {k: v for k, v in trial.model_dump().items() if (v not in [None, "", []])}
        missing = self.compute_missing_fields(trial)
        state = BeliefState(trial_profile=trial, known_fields=known, missing_fields=missing, ambiguities=[], evidence_items=[])
        suff, conf, rationale, evidence_gaps = self.estimate_evidence_sufficiency(trial, [], state)
        state.evidence_sufficiency = suff
        state.confidence = conf
        state.sufficiency_rationale = rationale
        state.evidence_gaps = evidence_gaps
        return state

    def compute_missing_fields(self, trial: TrialProfile) -> List[str]:
        important_fields = ["trial_id", "condition", "intervention", "sponsor", "phase", "status", "outcome_signal", "program_context"]
        return [f for f in important_fields if field_is_missing(getattr(trial, f, None))]

    def evidence_source_flags(self, evidence_items: List[EvidenceItem]) -> Dict[str, bool]:
        flags = {"registry": False, "publication": False, "sponsor": False}
        for ev in evidence_items:
            st = getattr(ev.source_type, "value", ev.source_type)
            if st in flags:
                flags[st] = True
        return flags

    def _effective_missing_fields(self, state: BeliefState) -> List[str]:
        missing = set(state.missing_fields)
        retrieval = state.retrieval_status
        if retrieval.get("publication") == "found" and not field_is_missing(state.trial_profile.outcome_signal):
            missing.discard("outcome_signal")
        if retrieval.get("sponsor") == "found" and not field_is_missing(state.trial_profile.program_context):
            missing.discard("program_context")
        return sorted(missing)

    def _best_sponsor_scope(self, state: BeliefState) -> Optional[str]:
        scopes = []
        for ev in state.evidence_items:
            if getattr(ev.source_type, "value", ev.source_type) == "sponsor":
                scope = (ev.extracted_fields or {}).get("sponsor_context_scope")
                if scope:
                    scopes.append(scope)
        if "trial_program_specific" in scopes:
            return "trial_program_specific"
        if "drug_level" in scopes:
            return "drug_level"
        if "unclear" in scopes:
            return "unclear"
        return None

    def _best_sponsor_reason(self, state: BeliefState) -> Optional[str]:
        for ev in state.evidence_items:
            if getattr(ev.source_type, "value", ev.source_type) == "sponsor":
                reason = (ev.extracted_fields or {}).get("sponsor_scope_reason")
                if reason:
                    return reason
        return None

    def _build_program_context_text(self, state: BeliefState, ev: EvidenceItem) -> Optional[str]:
        trial = state.trial_profile
        llm = ev.llm_semantic_signals or {}
        continuity = llm.get("program_continuity_signal")
        rationale = (llm.get("rationale") or "").strip()
        title = (trial.title or "").strip()
        phase = (trial.phase or "").strip()
        status = (trial.status or "").strip()
        sponsor = (trial.sponsor or "").strip()
        scope = (ev.extracted_fields or {}).get("sponsor_context_scope") or self._best_sponsor_scope(state)
        scope_reason = (ev.extracted_fields or {}).get("sponsor_scope_reason") or self._best_sponsor_reason(state) or rationale

        def _compact_title(t: str) -> str:
            t = t.strip()
            if not t:
                return ""
            return t if len(t) <= 140 else t[:137].rstrip() + "..."

        compact_title = _compact_title(title)
        phase_status = " ".join([x for x in [phase, status] if x]).strip()

        if scope == "trial_program_specific":
            if continuity == "continuity_present":
                if scope_reason:
                    if compact_title:
                        return f"Trial/program-specific sponsor evidence supports continued development for {compact_title}: {scope_reason}"
                    return f"Trial/program-specific sponsor evidence supports continued development: {scope_reason}"
                if compact_title and phase_status:
                    return f"Trial/program-specific sponsor evidence supports continued {phase_status.lower()} development for {compact_title}."
                return "Trial/program-specific sponsor evidence supports continued development."
            if continuity == "discontinuity_present":
                if scope_reason:
                    return f"Trial/program-specific sponsor evidence indicates potential discontinuity: {scope_reason}"
                if compact_title:
                    return f"Trial/program-specific sponsor evidence indicates potential discontinuity for {compact_title}."
                return "Trial/program-specific sponsor evidence indicates potential discontinuity."

        if scope == "drug_level":
            if continuity == "continuity_present":
                if sponsor and phase_status:
                    return f"Drug-level sponsor evidence suggests broader NSCLC continuity under {sponsor}, but it does not directly establish trial-specific program continuity for this {phase_status.lower()} study."
                return "Drug-level sponsor evidence suggests broader NSCLC continuity, but it does not directly establish trial-specific program continuity."
            if continuity == "discontinuity_present":
                return "Drug-level sponsor evidence suggests possible reprioritization at the drug or indication level, but it does not directly establish trial-specific discontinuity."
            return "Drug-level sponsor evidence was retrieved, but it does not directly establish trial-specific program continuity."

        if scope == "unclear":
            return "Sponsor-facing evidence was retrieved, but its relevance to the specific trial/program remains unclear."

        ctx, _ = extract_simple_program_context(ev.raw_text)
        return ctx

    def _recent_action_failures(self, state: BeliefState, action: ActionType, window: int = 3) -> int:
        recent = [h for h in state.action_history if h.get("selected_action") == action.value][-window:]
        failures = 0
        for h in recent:
            result = h.get("action_result") or {}
            status = str(result.get("status", "")).lower()
            retrieved_count = int(result.get("retrieved_count", 0) or 0)
            state_change = str(h.get("state_change", "")).lower()
            material_update = result.get("material_update")
            if (
                retrieved_count == 0
                or status in {"not_found", "failed", "no_trial_id", "no_change"}
                or state_change in {"no_new_evidence", "no_material_update"}
                or material_update is False
            ):
                failures += 1
        return failures

    def _recent_no_gain_actions(self, state: BeliefState, action: ActionType, window: int = 3) -> int:
        recent = [h for h in state.action_history if h.get("selected_action") == action.value][-window:]
        no_gain = 0
        for h in recent:
            if str(h.get("state_change", "")).lower() in {"no_new_evidence", "no_material_update"}:
                no_gain += 1
        return no_gain

    def _recent_material_updates(self, state: BeliefState, action: ActionType, window: int = 3) -> int:
        recent = [h for h in state.action_history if h.get("selected_action") == action.value][-window:]
        updates = 0
        for h in recent:
            result = h.get("action_result") or {}
            if result.get("material_update") is True or str(h.get("state_change", "")).lower() == "state_updated":
                updates += 1
        return updates

    def _action_blocked(self, state: BeliefState, action: ActionType) -> Tuple[bool, Optional[str]]:
        effective_missing = self._effective_missing_fields(state)

        if action == ActionType.RETRIEVE_PUBLICATION:
            failures = self._recent_action_failures(state, action, window=3)
            no_gain = self._recent_no_gain_actions(state, action, window=3)
            if failures >= 2 and field_is_missing(state.trial_profile.outcome_signal):
                return True, "publication retrieval already failed multiple times without resolving outcome evidence"
            if no_gain >= 1 and state.retrieval_status.get("publication") == "found" and field_is_missing(state.trial_profile.outcome_signal):
                return True, "publication retrieval already returned evidence but did not materially update the outcome state"
            return False, None

        if action == ActionType.RETRIEVE_SPONSOR_PAGE:
            failures = self._recent_action_failures(state, action, window=3)
            no_gain = self._recent_no_gain_actions(state, action, window=3)
            if failures >= 2 and field_is_missing(state.trial_profile.program_context):
                return True, "sponsor retrieval already failed multiple times without resolving program context"
            if no_gain >= 1 and state.retrieval_status.get("sponsor") == "found" and field_is_missing(state.trial_profile.program_context):
                return True, "sponsor retrieval already returned evidence but did not materially update the program-context state"
            return False, None

        if action == ActionType.QUERY_REGISTRY_DETAIL:
            failures = self._recent_action_failures(state, action, window=2)
            if failures >= 1 and (not effective_missing or set(effective_missing).issubset({"outcome_signal", "program_context"})):
                return True, "registry detail retry is unlikely to add value because remaining gaps are not registry-native"
            return False, None

        if action == ActionType.PERFORM_ALIAS_DISAMBIGUATION:
            failures = self._recent_action_failures(state, action, window=2)
            if failures >= 1:
                return True, "alias disambiguation already failed to produce a material update"
            return False, None

        return False, None

    def diagnose_highest_priority_gap(self, state: BeliefState) -> Tuple[str, str]:
        effective_missing = self._effective_missing_fields(state)
        if state.ambiguities:
            return "intervention identity ambiguity", "ambiguity blocks reliable downstream retrieval"
        if "outcome_signal" in effective_missing:
            return "outcome-related evidence", "publication-level evidence is still missing or has not materially resolved the outcome signal"
        if "program_context" in effective_missing:
            return "sponsor-side program context", "program-context evidence is still missing or has not materially resolved sponsor-side trajectory"
        if "intervention" in effective_missing or "sponsor" in effective_missing:
            return "intervention / sponsor identity", "formal entity identity should be stabilized first"
        if ("phase" in effective_missing or "status" in effective_missing) and "program_context" not in effective_missing:
            return "registry metadata refinement", "core structured metadata remains incomplete"

        state_summary = compact_state_summary(state)
        if self.llm_reasoner:
            out = self.llm_reasoner.diagnose_gap(state_summary)
            if out and out.get("highest_priority_gap"):
                return out["highest_priority_gap"], out.get("priority_reason", "LLM semantic diagnosis")
        return "no critical gap", "no material gap remains"

    def select_next_action(self, state: BeliefState, gap: Optional[str] = None) -> Tuple[ActionType, str]:
        gap = gap or self.diagnose_highest_priority_gap(state)[0]
        if state.evidence_sufficiency == EvidenceSufficiency.SUFFICIENT:
            return ActionType.STOP, "evidence already sufficient"

        effective_missing = self._effective_missing_fields(state)
        if gap == "sponsor-side program context" or ("program_context" in effective_missing and "outcome_signal" not in effective_missing):
            blocked, reason = self._action_blocked(state, ActionType.RETRIEVE_SPONSOR_PAGE)
            if not blocked:
                return ActionType.RETRIEVE_SPONSOR_PAGE, "program-context gap should go directly to sponsor-facing evidence before retrying registry metadata"
            # Prefer stopping over bouncing back to registry when sponsor evidence has already been tried without material gain.
            if state.retrieval_status.get("sponsor") in {"found", "not_found"}:
                return ActionType.STOP, reason or "sponsor-facing evidence has already been considered and further registry retries are unlikely to resolve program context"
            if ("phase" in effective_missing or "status" in effective_missing) and self._recent_no_gain_actions(state, ActionType.QUERY_REGISTRY_DETAIL, window=1) == 0:
                reg_blocked, reg_reason = self._action_blocked(state, ActionType.QUERY_REGISTRY_DETAIL)
                if not reg_blocked:
                    return ActionType.QUERY_REGISTRY_DETAIL, "sponsor retrieval appears exhausted; try one registry refinement pass for remaining structured metadata"
                return ActionType.STOP, reg_reason or reason or "program-context retrieval no longer adds value"
            return ActionType.STOP, reason or "program-context retrieval no longer adds value"

        action_space = [a.value for a in ActionType]
        state_summary = compact_state_summary(state)
        selected_action: Optional[ActionType] = None
        selected_reason: Optional[str] = None
        if self.llm_reasoner:
            out = self.llm_reasoner.choose_action(state_summary=state_summary, action_space=action_space, gap_diagnosis={"highest_priority_gap": gap})
            if out and out.get("selected_action") in action_space:
                selected_action = ActionType(out["selected_action"])
                selected_reason = out.get("reason", "LLM source-selection reasoning")

        if selected_action is None:
            if gap == "intervention identity ambiguity":
                selected_action, selected_reason = ActionType.PERFORM_ALIAS_DISAMBIGUATION, "ambiguity must be resolved before further evidence retrieval"
            elif gap == "outcome-related evidence":
                selected_action, selected_reason = ActionType.RETRIEVE_PUBLICATION, "highest information gain for outcome uncertainty"
            elif gap == "sponsor-side program context":
                selected_action, selected_reason = ActionType.RETRIEVE_SPONSOR_PAGE, "best source for program continuity and sponsor context"
            elif gap == "intervention / sponsor identity":
                selected_action, selected_reason = ActionType.QUERY_REGISTRY_DETAIL, "registry detail most reliable for formal metadata"
            elif gap == "registry metadata refinement":
                if "program_context" in effective_missing and state.retrieval_status.get("sponsor") != "found":
                    selected_action, selected_reason = ActionType.RETRIEVE_SPONSOR_PAGE, "remaining missing program-context signal should be handled by sponsor-facing evidence before another registry retry"
                elif "outcome_signal" in effective_missing and state.retrieval_status.get("publication") == "found":
                    if state.retrieval_status.get("sponsor") != "found":
                        selected_action, selected_reason = ActionType.RETRIEVE_SPONSOR_PAGE, "publication was already tried without resolving outcome; sponsor evidence is more useful than another registry retry"
                    else:
                        selected_action, selected_reason = ActionType.STOP, "another registry retry is unlikely to resolve the remaining non-registry evidence gaps"
                else:
                    selected_action, selected_reason = ActionType.QUERY_REGISTRY_DETAIL, "formal structured source is preferred"
            else:
                selected_action, selected_reason = ActionType.STOP, "no important gap remains"

        blocked, block_reason = self._action_blocked(state, selected_action)
        if not blocked:
            return selected_action, selected_reason or "selected action"

        if selected_action == ActionType.RETRIEVE_PUBLICATION:
            if field_is_missing(state.trial_profile.program_context) and state.retrieval_status.get("sponsor") != "found":
                return ActionType.RETRIEVE_SPONSOR_PAGE, f"switching away from repeated no-gain publication retrieval: {block_reason}"
            return ActionType.STOP, f"stopping because repeated publication retrieval is no longer adding value: {block_reason}"
        if selected_action == ActionType.RETRIEVE_SPONSOR_PAGE:
            if field_is_missing(state.trial_profile.outcome_signal) and state.retrieval_status.get("publication") != "found":
                return ActionType.RETRIEVE_PUBLICATION, f"switching away from repeated no-gain sponsor retrieval: {block_reason}"
            return ActionType.STOP, f"stopping because repeated sponsor retrieval is no longer adding value: {block_reason}"
        if selected_action == ActionType.QUERY_REGISTRY_DETAIL:
            return ActionType.STOP, f"stopping because registry retry is unlikely to add value: {block_reason}"
        return ActionType.STOP, block_reason or "action blocked"

    def perform_action(self, state: BeliefState, action: ActionType) -> Tuple[List[EvidenceItem], Dict[str, object]]:
        trial = state.trial_profile
        meta: Dict[str, object] = {"retrieved_count": 0}
        if action == ActionType.QUERY_REGISTRY_DETAIL:
            state.retrieval_status["registry"] = "attempted"
            if not trial.trial_id:
                state.retrieval_status["registry"] = "failed"
                meta["status"] = "no_trial_id"
                return [], meta
            ev = self.registry_client.get_trial_detail(trial.trial_id)
            state.retrieval_status["registry"] = "found"
            meta["retrieved_count"] = 1
            meta["status"] = "found"
            return [ev], meta
        if action == ActionType.RETRIEVE_PUBLICATION:
            state.retrieval_status["publication"] = "attempted"
            evs = self.pubmed_client.retrieve_publications(trial)
            state.retrieval_status["publication"] = "found" if evs else "not_found"
            meta["retrieved_count"] = len(evs)
            meta["status"] = state.retrieval_status["publication"]
            if evs:
                meta["queries"] = list({ev.extracted_fields.get("retrieval_query") for ev in evs if ev.extracted_fields.get("retrieval_query")})
            return evs, meta
        if action == ActionType.RETRIEVE_SPONSOR_PAGE:
            state.retrieval_status["sponsor"] = "attempted"
            evs = self.sponsor_client.retrieve_sponsor_evidence(trial)
            state.retrieval_status["sponsor"] = "found" if evs else "not_found"
            meta["retrieved_count"] = len(evs)
            meta["status"] = state.retrieval_status["sponsor"]
            if evs:
                meta["urls"] = [ev.url for ev in evs if ev.url]
            return evs, meta
        if action == ActionType.PERFORM_ALIAS_DISAMBIGUATION:
            intervention = trial.intervention or ""
            if intervention and "keytruda" in safe_lower(intervention) and "pembrolizumab" not in safe_lower(intervention):
                return [
                    EvidenceItem(
                        source_type="registry",
                        source_name="Alias Disambiguation Module",
                        raw_text="Alias disambiguation result: KEYTRUDA == pembrolizumab",
                        extracted_fields={"intervention": "pembrolizumab"},
                        confidence=0.8,
                        url=None,
                    )
                ], {"retrieved_count": 1, "status": "resolved"}
            return [], {"retrieved_count": 0, "status": "no_change"}
        return [], meta

    def enrich_evidence_with_llm(self, state: BeliefState, evidence_items: List[EvidenceItem]) -> None:
        if not self.llm_reasoner:
            return
        trial_profile_dict = state.trial_profile.model_dump()
        for ev in evidence_items:
            out = self.llm_reasoner.extract_semantic_signals(trial_profile=trial_profile_dict, source_type=getattr(ev.source_type, "value", ev.source_type), raw_text=ev.raw_text)
            if not out:
                continue
            ev.llm_semantic_signals = out
            if "drug_aliases" in out and out["drug_aliases"] and "drug_aliases" not in ev.extracted_fields:
                ev.extracted_fields["drug_aliases"] = out["drug_aliases"]

    def update_trial_profile_from_evidence(self, state: BeliefState, evidence_items: List[EvidenceItem]) -> None:
        for ev in evidence_items:
            source_type = getattr(ev.source_type, "value", ev.source_type)
            llm = ev.llm_semantic_signals or {}
            if source_type == "publication" and field_is_missing(state.trial_profile.outcome_signal):
                llm_signal = llm.get("outcome_signal")
                if llm_signal in {"Positive", "Negative", "Mixed"}:
                    state.trial_profile.outcome_signal = llm_signal
                else:
                    sig, _ = extract_simple_outcome_signal(ev.raw_text)
                    if sig is not None:
                        state.trial_profile.outcome_signal = sig
            if source_type == "sponsor":
                scope = (ev.extracted_fields or {}).get("sponsor_context_scope")
                if scope:
                    state.sponsor_context_scope = scope
                if field_is_missing(state.trial_profile.program_context):
                    ctx = self._build_program_context_text(state, ev)
                    if ctx is not None:
                        state.trial_profile.program_context = ctx

    def update_state_with_evidence(self, state: BeliefState, new_evidence: List[EvidenceItem]) -> None:
        self.enrich_evidence_with_llm(state, new_evidence)
        self.update_trial_profile_from_evidence(state, new_evidence)

        for ev in new_evidence:
            state.evidence_items.append(ev)
            for k, v in ev.extracted_fields.items():
                if v in [None, ""]:
                    continue
                if hasattr(state.trial_profile, k):
                    current_v = getattr(state.trial_profile, k)
                    if field_is_missing(current_v):
                        setattr(state.trial_profile, k, v)
                    elif str(current_v).strip().lower() != str(v).strip().lower():
                        conflict = f"Conflict on {k}: existing={current_v} vs new={v}"
                        if conflict not in state.ambiguities:
                            state.ambiguities.append(conflict)
            if ev.url:
                state.trial_profile.linked_sources.append(ev.url)

        state.trial_profile.linked_sources = list(dict.fromkeys(state.trial_profile.linked_sources))
        state.known_fields = {k: v for k, v in state.trial_profile.model_dump().items() if (v not in [None, "", []])}
        state.missing_fields = self.compute_missing_fields(state.trial_profile)
        suff, conf, rationale, evidence_gaps = self.estimate_evidence_sufficiency(state.trial_profile, state.evidence_items, state)
        state.evidence_sufficiency = suff
        state.confidence = conf
        state.sufficiency_rationale = rationale
        state.evidence_gaps = evidence_gaps

    def estimate_evidence_sufficiency(self, trial: TrialProfile, evidence_items: List[EvidenceItem], state: Optional[BeliefState] = None):
        has_registry_core = all([
            not field_is_missing(trial.trial_id),
            not field_is_missing(trial.condition),
            not field_is_missing(trial.intervention),
            not field_is_missing(trial.sponsor),
            not field_is_missing(trial.phase),
            not field_is_missing(trial.status),
        ])
        source_flags = self.evidence_source_flags(evidence_items)
        has_publication = source_flags["publication"]
        has_sponsor = source_flags["sponsor"]
        sponsor_scope = self._best_sponsor_scope(state) if state is not None else None
        has_trial_specific_sponsor = sponsor_scope == "trial_program_specific"
        has_outcome = not field_is_missing(trial.outcome_signal)
        has_program_context = not field_is_missing(trial.program_context)

        retrieval_status = state.retrieval_status if state is not None else {"publication": "not_attempted", "sponsor": "not_attempted", "registry": "not_attempted"}
        evidence_gaps: List[str] = []
        if retrieval_status.get("publication") == "not_attempted":
            evidence_gaps.append("Publication retrieval has not been attempted yet.")
        elif retrieval_status.get("publication") == "not_found":
            evidence_gaps.append("Publication retrieval was attempted but no sufficiently relevant PubMed evidence was found.")
        if not has_outcome:
            evidence_gaps.append("Outcome signal is still unresolved.")
        if retrieval_status.get("sponsor") == "not_attempted":
            evidence_gaps.append("Sponsor-facing evidence retrieval has not been attempted yet.")
        elif retrieval_status.get("sponsor") == "not_found":
            evidence_gaps.append("Sponsor-facing retrieval was attempted but no usable official-page evidence was found.")
        elif retrieval_status.get("sponsor") == "found" and sponsor_scope == "drug_level":
            evidence_gaps.append("Sponsor evidence is currently drug-level rather than trial/program-specific.")
        elif retrieval_status.get("sponsor") == "found" and sponsor_scope == "unclear":
            evidence_gaps.append("Sponsor evidence was retrieved but its relevance to the specific trial/program remains unclear.")
        if not has_program_context:
            evidence_gaps.append("Program context remains unresolved.")

        if has_registry_core and has_publication and has_trial_specific_sponsor and has_outcome and has_program_context:
            suff = EvidenceSufficiency.SUFFICIENT
        elif has_registry_core and ((has_publication and has_outcome) or (has_sponsor and has_program_context)):
            suff = EvidenceSufficiency.PARTIAL
        else:
            suff = EvidenceSufficiency.INSUFFICIENT

        rationale = "Evidence sufficiency is based on both field coverage and cross-source support, not just filled registry metadata."
        conf = 0.20
        if has_registry_core:
            conf += 0.18
        if retrieval_status.get("publication") == "found":
            conf += 0.08
        if retrieval_status.get("sponsor") == "found":
            conf += 0.03 if sponsor_scope == "drug_level" else 0.06
        if has_publication and has_outcome:
            conf += 0.18
        if has_sponsor and has_program_context:
            conf += 0.06 if sponsor_scope == "drug_level" else 0.12
        if len(evidence_items) >= 2:
            conf += 0.04

        if self.llm_reasoner and state is not None:
            out = self.llm_reasoner.judge_sufficiency(compact_state_summary(state), budget_remaining=max(0, 999 - state.step_count))
            if out and out.get("evidence_sufficiency") in {e.value for e in EvidenceSufficiency}:
                suff = EvidenceSufficiency(out["evidence_sufficiency"])
                rationale = out.get("rationale", rationale)
                conf = max(conf, float(out.get("confidence", conf) or conf))

        if suff == EvidenceSufficiency.INSUFFICIENT:
            conf = min(conf, 0.55)
        elif suff == EvidenceSufficiency.PARTIAL:
            conf = min(conf, 0.78)
        else:
            conf = min(conf, 0.93)
        return suff, round(clip(conf, 0.0, 0.99), 3), rationale, evidence_gaps

    def should_stop(self, state: BeliefState, budget: int):
        if state.step_count >= budget:
            return True, "budget exhausted"
        if state.evidence_sufficiency == EvidenceSufficiency.SUFFICIENT:
            return True, state.sufficiency_rationale or "evidence sufficient"
        if state.retrieval_status.get("publication") == "not_found" and state.retrieval_status.get("sponsor") == "not_found":
            return True, "publication and sponsor retrieval were both attempted without usable evidence"
        if self._recent_no_gain_actions(state, ActionType.RETRIEVE_PUBLICATION, window=2) >= 2 and field_is_missing(state.trial_profile.outcome_signal):
            if state.retrieval_status.get("sponsor") == "found" or self._recent_no_gain_actions(state, ActionType.RETRIEVE_SPONSOR_PAGE, window=1) >= 1:
                return True, "repeated publication retrieval produced no material update after alternative evidence sources were considered"
        if self._recent_action_failures(state, ActionType.RETRIEVE_PUBLICATION, window=3) >= 2 and self._recent_action_failures(state, ActionType.RETRIEVE_SPONSOR_PAGE, window=3) >= 1 and field_is_missing(state.trial_profile.outcome_signal) and field_is_missing(state.trial_profile.program_context):
            return True, "multiple no-gain publication attempts plus sponsor failure indicate diminishing returns under current budget"
        if (not field_is_missing(state.trial_profile.outcome_signal)) and (not field_is_missing(state.trial_profile.program_context)):
            if self._recent_no_gain_actions(state, ActionType.QUERY_REGISTRY_DETAIL, window=2) >= 1 or self._recent_no_gain_actions(state, ActionType.PERFORM_ALIAS_DISAMBIGUATION, window=2) >= 1:
                return True, "key outcome and program-context fields are already resolved; additional no-gain refinement steps are not adding value"
        return False, "continue"

    def synthesize_momentum(self, state: BeliefState) -> Dict[str, object]:
        if self.llm_reasoner and state.evidence_sufficiency != EvidenceSufficiency.INSUFFICIENT:
            out = self.llm_reasoner.synthesize_high_level(compact_state_summary(state))
            if out and out.get("momentum") in {m.value for m in MomentumLabel}:
                conf = float(out.get("confidence_estimate", state.confidence) or state.confidence)
                if state.evidence_sufficiency == EvidenceSufficiency.PARTIAL:
                    conf = min(conf, 0.78)
                return {
                    "label": out["momentum"],
                    "confidence": round(clip(conf, 0.0, 0.95), 3),
                    "rationale": out.get("brief_rationale", "LLM high-level synthesis"),
                    "evidence_sufficiency_note": state.evidence_sufficiency.value,
                }

        trial = state.trial_profile
        status = safe_lower(trial.status)
        positive_signal = safe_lower(trial.outcome_signal) == "positive"
        negative_signal = safe_lower(trial.outcome_signal) == "negative"
        ctx = safe_lower(trial.program_context)

        label = MomentumLabel.UNCERTAIN
        rationale = "Current public evidence is limited or mixed, so the overall trajectory remains uncertain."
        score = 0.40 if state.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT else 0.55

        if positive_signal and "continuity signal present" in ctx and status in {"completed", "recruiting", "active, not recruiting"}:
            label = MomentumLabel.ADVANCING
            rationale = "Positive outcome evidence together with sponsor-side continuity supports an advancing trajectory."
            score = 0.72
        elif negative_signal and "discontinuity signal present" in ctx:
            label = MomentumLabel.STALLED
            rationale = "Negative outcome evidence together with sponsor-side discontinuity supports a stalled trajectory."
            score = 0.70
        elif status == "recruiting":
            label = MomentumLabel.UNCERTAIN
            rationale = "The study is still recruiting and no outcome evidence is available yet, so trajectory remains uncertain."
            score = 0.42
        elif status == "terminated":
            label = MomentumLabel.UNCERTAIN
            rationale = "The study is terminated, but the current evidence does not establish a reliable efficacy or program-level interpretation."
            score = 0.42

        if state.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT:
            score = min(score, 0.58)
        elif state.evidence_sufficiency == EvidenceSufficiency.PARTIAL:
            score = min(score, 0.78)

        return {"label": label.value, "confidence": round(clip(score, 0.0, 0.95), 3), "rationale": rationale, "evidence_sufficiency_note": state.evidence_sufficiency.value}

    def build_final_output(self, state: BeliefState) -> TrialIntelligenceOutput:
        momentum = self.synthesize_momentum(state)
        profile_dict = {
            "trial_id": state.trial_profile.trial_id,
            "title": state.trial_profile.title,
            "condition": state.trial_profile.condition,
            "intervention": state.trial_profile.intervention,
            "sponsor": state.trial_profile.sponsor,
            "phase": state.trial_profile.phase,
            "status": state.trial_profile.status,
            "brief_summary": state.trial_profile.brief_summary,
            "outcome_signal": state.trial_profile.outcome_signal,
            "program_context": state.trial_profile.program_context,
            "linked_sources": state.trial_profile.linked_sources,
            "retrieval_status": state.retrieval_status,
            "evidence_summary": [ev.model_dump() for ev in state.evidence_items],
            "sufficiency_rationale": state.sufficiency_rationale,
        }
        return TrialIntelligenceOutput(
            structured_trial_profile=profile_dict,
            unresolved_fields=state.missing_fields,
            evidence_gaps=state.evidence_gaps,
            ambiguities=state.ambiguities,
            evidence_sufficiency=state.evidence_sufficiency.value,
            confidence=round(state.confidence, 3),
            momentum=momentum,
            reasoning_trace=state.action_history,
        )

    def run_for_one_trial(self, trial: TrialProfile, budget: int = 5) -> TrialIntelligenceOutput:
        state = self.initialize_belief_state(trial)
        while True:
            stop, stop_reason = self.should_stop(state, budget)
            if stop:
                state.stop_reason = stop_reason
                state.action_history.append({
                    "step": state.step_count,
                    "decision": "Stop",
                    "reason": stop_reason,
                    "evidence_sufficiency": state.evidence_sufficiency.value,
                    "confidence": round(state.confidence, 3),
                    "evidence_gaps": state.evidence_gaps,
                    "retrieval_status": dict(state.retrieval_status),
                })
                break

            gap, gap_reason = self.diagnose_highest_priority_gap(state)
            action, action_reason = self.select_next_action(state, gap)
            history = {
                "step": state.step_count + 1,
                "known_fields": list(state.known_fields.keys()),
                "missing_fields": state.missing_fields.copy(),
                "evidence_gaps": state.evidence_gaps.copy(),
                "retrieval_status": dict(state.retrieval_status),
                "ambiguities": state.ambiguities.copy(),
                "evidence_sufficiency": state.evidence_sufficiency.value,
                "confidence": round(state.confidence, 3),
                "gap_diagnosis": gap,
                "gap_reason": gap_reason,
                "selected_action": action.value,
                "action_reason": action_reason,
            }
            if action == ActionType.STOP:
                state.stop_reason = action_reason
                state.action_history.append(history)
                break
            new_evidence, action_meta = self.perform_action(state, action)
            history["action_result"] = action_meta
            before_known = set(state.known_fields.keys())
            before_outcome = state.trial_profile.outcome_signal
            before_context = state.trial_profile.program_context
            before_evidence_count = len(state.evidence_items)
            state.action_history.append(history)
            self.update_state_with_evidence(state, new_evidence)
            after_known = set(state.known_fields.keys())
            updated_fields = [
                field_name for field_name, before_val, after_val in [
                    ("outcome_signal", before_outcome, state.trial_profile.outcome_signal),
                    ("program_context", before_context, state.trial_profile.program_context),
                ] if before_val != after_val
            ]
            registry_structural_gain = bool((after_known - before_known) & {"trial_id", "condition", "intervention", "sponsor", "phase", "status"})
            material_update = bool(updated_fields) or registry_structural_gain
            if not new_evidence and action in {ActionType.RETRIEVE_PUBLICATION, ActionType.RETRIEVE_SPONSOR_PAGE, ActionType.QUERY_REGISTRY_DETAIL}:
                history["state_change"] = "no_new_evidence"
            elif not material_update:
                history["state_change"] = "no_material_update"
                if history.get("action_result", {}).get("status") == "found":
                    history["action_result"]["status"] = "no_change"
            else:
                history["state_change"] = "state_updated"
            history["action_result"]["material_update"] = material_update
            history["action_result"]["new_evidence_count"] = max(0, len(state.evidence_items) - before_evidence_count)
            history["action_result"]["updated_fields"] = updated_fields
            state.step_count += 1
        return self.build_final_output(state)

    def run(self, query: QueryInput) -> List[TrialIntelligenceOutput]:
        candidates = self.discover_candidate_trials(query)
        return [self.run_for_one_trial(trial, budget=query.budget) for trial in candidates]
