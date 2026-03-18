from __future__ import annotations

import argparse
import json
import sys
from typing import List

from agent import TrialIntelligenceAgent
from clients import ClinicalTrialsClient, PubMedClient, SponsorPageClient
from llm_reasoner import LLMReasoner
from schemas import QueryInput, TrialIntelligenceOutput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical Trial Intelligence Agent")
    parser.add_argument("--disease", required=True, help="Disease query, e.g. 'non-small cell lung cancer'")
    parser.add_argument("--drug", default=None, help="Drug query, e.g. 'pembrolizumab'")
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--sponsor-config", default=None, help="Path to sponsor source JSON config")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--llm-mode", choices=["off", "optional", "required"], default="optional")
    parser.add_argument("--ollama-model", default=None, help="Override OLLAMA_MODEL")
    parser.add_argument("--ollama-base-url", default=None, help="Override OLLAMA_BASE_URL")
    parser.add_argument("--ollama-think", action="store_true", help="Ask Ollama for thinking-capable responses when supported")
    parser.add_argument("--ollama-keep-alive", default=None, help="Override OLLAMA_KEEP_ALIVE, e.g. 5m")
    return parser


def serialize(outputs: List[TrialIntelligenceOutput], pretty: bool = False) -> str:
    data = [o.model_dump() for o in outputs]
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def main() -> int:
    args = build_parser().parse_args()

    query = QueryInput(
        disease=args.disease,
        drug=args.drug,
        max_trials=args.max_trials,
        budget=args.budget,
    )

    llm_reasoner = None
    if args.llm_mode != "off":
        llm_reasoner = LLMReasoner(
            enabled=True,
            required=(args.llm_mode == "required"),
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            think=args.ollama_think,
            keep_alive=args.ollama_keep_alive,
        )

    agent = TrialIntelligenceAgent(
        registry_client=ClinicalTrialsClient(),
        pubmed_client=PubMedClient(),
        sponsor_client=SponsorPageClient(args.sponsor_config),
        llm_reasoner=llm_reasoner,
    )

    try:
        outputs = agent.run(query)
        print(serialize(outputs, pretty=args.pretty))
        return 0
    except Exception as e:
        err = {"error": str(e)}
        print(json.dumps(err, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
