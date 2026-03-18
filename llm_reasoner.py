from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMReasoner:
    def __init__(
        self,
        enabled: bool = True,
        required: bool = False,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        think: bool = False,
        keep_alive: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.enabled = enabled
        self.required = required
        self.model = model or os.getenv("OLLAMA_MODEL", "")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.think = think
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        self.timeout = timeout or float(os.getenv("REQUEST_TIMEOUT", "120"))
        self.chat_url = f"{self.base_url}/api/chat"
        self.available = False

        if self.enabled and self.model:
            self.available = self._check_server()

        if self.enabled and self.required and not self.available:
            raise RuntimeError(
                "LLM mode is required, but Ollama is not reachable or OLLAMA_MODEL is not configured. "
                f"Tried {self.chat_url} with model '{self.model}'."
            )

    @property
    def is_available(self) -> bool:
        return self.available and bool(self.model)

    def _check_server(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout, 10))
            return r.ok
        except Exception:
            return False

    def maybe_call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if not self.is_available:
            if self.required:
                raise RuntimeError(
                    "LLM mode is required, but Ollama server/model is not configured correctly. "
                    f"base_url={self.base_url}, model={self.model}"
                )
            return None

        content = (
            "Return ONLY a single valid JSON object.\n"
            "Do not include markdown fences.\n"
            "Do not include any explanation before or after the JSON.\n\n"
            f"Schema hint:\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n\n"
            f"Task:\n{user_prompt}"
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
            "keep_alive": self.keep_alive,
        }
        if self.think:
            payload["think"] = True

        try:
            response = requests.post(self.chat_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception as e:
            if self.required:
                raise RuntimeError(f"Ollama request failed: {e}") from e
            return None

        data = response.json()
        text = self._extract_text(data).strip()
        if not text:
            if self.required:
                raise RuntimeError("LLM returned empty output.")
            return None

        parsed = self._extract_json_object(text)
        if parsed is None and self.required:
            raise RuntimeError(f"LLM did not return valid JSON. Raw output: {text[:1000]}")
        return parsed

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        if isinstance(data.get("message"), dict):
            content = data["message"].get("content", "")
            if isinstance(content, str):
                return content
        if isinstance(data.get("response"), str):
            return data["response"]
        return ""

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
        raw = (raw or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        m = re.search(r"(\{.*\})", raw, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return None

    def extract_semantic_signals(self, trial_profile: Dict[str, Any], source_type: str, raw_text: str) -> Optional[Dict[str, Any]]:
        schema = {
            "outcome_signal": "Positive | Negative | Mixed | Unknown",
            "efficacy_cues": ["string"],
            "safety_cues": ["string"],
            "drug_aliases": ["string"],
            "program_continuity_signal": "continuity_present | discontinuity_present | unclear",
            "uncertainty_notes": ["string"],
            "rationale": "string",
        }
        system = (
            "You are a biomedical trial-intelligence extraction module. "
            "Use only explicit evidence from the provided text. "
            "If the evidence is weak, indirect, or absent, return outcome_signal='Unknown' and program_continuity_signal='unclear'. "
            "A broad drug indication page is drug-level evidence, not trial-specific program evidence."
        )
        user = (
            f"Trial profile:\n{json.dumps(trial_profile, ensure_ascii=False)}\n\n"
            f"Source type: {source_type}\n"
            f"Evidence text:\n{raw_text[:12000]}\n\n"
            "Be conservative. Registry metadata alone is usually insufficient for outcome claims."
        )
        parsed = self.maybe_call_json(system_prompt=system, user_prompt=user, schema_hint=schema)
        if parsed is None:
            return None
        return {
            "outcome_signal": parsed.get("outcome_signal", "Unknown"),
            "efficacy_cues": parsed.get("efficacy_cues", []) or [],
            "safety_cues": parsed.get("safety_cues", []) or [],
            "drug_aliases": parsed.get("drug_aliases", []) or [],
            "program_continuity_signal": parsed.get("program_continuity_signal", "unclear"),
            "uncertainty_notes": parsed.get("uncertainty_notes", []) or [],
            "rationale": parsed.get("rationale", ""),
        }

    def diagnose_gap(self, state_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schema = {
            "highest_priority_gap": "string",
            "priority_reason": "string",
            "suggested_target_field": "string",
        }
        system = (
            "You are the missing-field reasoning module for a clinical-trial intelligence agent. "
            "Prefer outcome/publication gaps and sponsor-context gaps over already-filled registry metadata when evidence is insufficient."
        )
        user = f"Belief state summary:\n{json.dumps(state_summary, ensure_ascii=False, indent=2)}"
        parsed = self.maybe_call_json(system_prompt=system, user_prompt=user, schema_hint=schema)
        if parsed is None:
            return None
        return {
            "highest_priority_gap": parsed.get("highest_priority_gap", "outcome-related evidence"),
            "priority_reason": parsed.get("priority_reason", "Need more evidence to reduce uncertainty."),
            "suggested_target_field": parsed.get("suggested_target_field", "outcome_signal"),
        }

    def choose_action(self, state_summary: Dict[str, Any], action_space: List[str], gap_diagnosis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schema = {
            "selected_action": "one of action_space",
            "reason": "string",
            "expected_information_gain": "low | medium | high",
        }
        system = (
            "You are the source-selection reasoning module for a clinical-trial intelligence agent. "
            "Choose only one action from action_space. Be conservative and practical."
        )
        user = (
            f"Action space: {action_space}\n\n"
            f"Gap diagnosis:\n{json.dumps(gap_diagnosis, ensure_ascii=False, indent=2)}\n\n"
            f"Belief state summary:\n{json.dumps(state_summary, ensure_ascii=False, indent=2)}"
        )
        parsed = self.maybe_call_json(system_prompt=system, user_prompt=user, schema_hint=schema)
        if parsed is None:
            return None
        selected = parsed.get("selected_action", action_space[0])
        if selected not in action_space:
            selected = action_space[0]
        return {
            "selected_action": selected,
            "reason": parsed.get("reason", "Choose the next action that most reduces uncertainty."),
            "expected_information_gain": parsed.get("expected_information_gain", "medium"),
        }

    def judge_sufficiency(self, state_summary: Dict[str, Any], budget_remaining: int) -> Optional[Dict[str, Any]]:
        schema = {
            "evidence_sufficiency": "Insufficient | Partial | Sufficient",
            "continue_retrieval": True,
            "rationale": "string",
            "confidence": 0.0,
        }
        system = (
            "You are the evidence sufficiency and stopping module for a clinical-trial intelligence agent. "
            "Be conservative. If outcome evidence or sponsor context is missing, do not mark Sufficient."
        )
        user = (
            f"Budget remaining: {budget_remaining}\n\n"
            f"Belief state summary:\n{json.dumps(state_summary, ensure_ascii=False, indent=2)}"
        )
        parsed = self.maybe_call_json(system_prompt=system, user_prompt=user, schema_hint=schema)
        if parsed is None:
            return None
        return {
            "evidence_sufficiency": parsed.get("evidence_sufficiency", "Insufficient"),
            "continue_retrieval": bool(parsed.get("continue_retrieval", True)),
            "rationale": parsed.get("rationale", "Current public evidence is still limited."),
            "confidence": float(parsed.get("confidence", 0.5) or 0.5),
        }

    def synthesize_high_level(self, state_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        schema = {
            "outcome_signal": "Positive | Negative | Mixed | Unknown",
            "confidence_estimate": 0.0,
            "momentum": "Advancing | Uncertain | Stalled",
            "brief_rationale": "string",
        }
        system = (
            "You are the high-level interpretive synthesis module for a clinical-trial intelligence agent. "
            "Be bounded and conservative. If evidence is insufficient, prefer momentum='Uncertain'. Recruiting/completed alone does not imply Advancing or Stalled. Drug-level sponsor continuity does not by itself establish trial-specific program continuity."
        )
        user = f"Belief state summary:\n{json.dumps(state_summary, ensure_ascii=False, indent=2)}"
        parsed = self.maybe_call_json(system_prompt=system, user_prompt=user, schema_hint=schema)
        if parsed is None:
            return None
        return {
            "outcome_signal": parsed.get("outcome_signal", "Unknown"),
            "confidence_estimate": float(parsed.get("confidence_estimate", 0.5) or 0.5),
            "momentum": parsed.get("momentum", "Uncertain"),
            "brief_rationale": parsed.get("brief_rationale", "Evidence remains limited or mixed."),
        }
