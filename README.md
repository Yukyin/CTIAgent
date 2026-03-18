# CTIAgent: Active Information Acquisition for Clinical Trial Intelligence

CTIAgent is an agent for clinical trial intelligence built around three core ideas:

1. Multi-source evidence collection from ClinicalTrials.gov, PubMed, and sponsor-facing public pages.
2. Trial-centered reasoning with a belief state, so the system can decide what is still missing and which source to query next.
3. Conservative evidence handling, so that retrieving a source is not automatically treated as having sufficient support for a conclusion.



## What the CTIAgent is supposed to do

The CTIAgent is designed to:

1. initialize a trial-centered state from registry metadata
2. identify the most important missing field
3. choose the next source based on expected information gain
4. update the state only when evidence causes a material update
5. stop when further retrieval is unlikely to help

The main goal is to show an uncertainty-aware CTIAgent should be more careful than a fixed baseline when working with messy, incomplete public clinical trial evidence, such as:

- a trial is still recruiting
- publications are related but not directly outcome-confirming for the current NCT
- sponsor evidence is only drug-level rather than trial-specific




## Repository structure

- `main.py` — main CLI entry point for the agent
- `agent.py` — belief state, action selection, stopping logic, synthesis
- `clients.py` — retrieval clients for ClinicalTrials.gov, PubMed, and sponsor pages
- `llm_reasoner.py` — Ollama-backed reasoning and extraction helpers
- `schemas.py` — typed schemas and data models
- `utils.py` — helper functions
- `baseline_pipeline.py` — primitive fixed-order baseline
- `compare_agent_vs_baseline.py` — compare agent output vs baseline output
- `sponsor_sources.json` — sponsor-page URL configuration
- `results/` — example agent, baseline, and comparison outputs
- `docs/` — project documentation



## Runtime setup

```bash
pip install -r requirements.txt

export PATH=$WORK/ollama/bin:$PATH
export OLLAMA_MODELS=$WORK/ollama_models
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b
export OLLAMA_KEEP_ALIVE=30m
export REQUEST_TIMEOUT=600

nohup ollama serve > $WORK/ollama/ollama.log 2>&1 &
sleep 5
curl http://localhost:11434/api/tags

ollama pull gpt-oss:20b
ollama run gpt-oss:20b "ok"
```

### Notes

- Get Internet access for ClinicalTrials.gov, PubMed, and sponsor-page retrieval

- Start the Ollama server first, then pull the model, then run a short test.

  

## Quick start

### 1) Run the CTIAgent


```bash
python main.py \
  --disease "non-small cell lung cancer" \
  --drug "pembrolizumab" \
  --sponsor-config sponsor_sources.json \
  --max-trials 5 \
  --budget 5 \
  --llm-mode required \
  --ollama-model gpt-oss:20b \
  --ollama-base-url http://localhost:11434 \
  --pretty > results/agent_output.json
```

### 2) Run the baseline

The baseline uses a fixed order:  registry→publication→sponsor page→naive merge. 

It does not use belief state, source selection, stopping reasoning, or conservative material-update logic.

```bash
python baseline_pipeline.py \
  --disease "non-small cell lung cancer" \
  --drug "pembrolizumab" \
  --sponsor-config sponsor_sources.json \
  --max-trials 5 \
  --pretty > results/baseline_output.json
```

### 3) Compare CTIAgent with baseline

```bash
python compare_agent_vs_baseline.py \
  --agent-json agent_output.json \
  --baseline-json baseline_output.json \
  --pretty > results/compare_output.json
```

This comparison is most useful for checking:

- whether the CTIAgent is more conservative on incomplete trials
- whether the CTIAgent avoids over-interpreting drug-level sponsor pages
- whether the CTIAgent distinguishes retrieved evidence from material state updates




## Example outputs

After running the commands above, the main reference files are:

- `agent_output.json` — CTIAgent results
- `baseline_output.json` — Baseline results
- `compare_output.json` — Side-by-side comparison between CTIAgent and baseline

Typical things to inspect in `compare_output.json`:

- `outcome_signal`
- `program_context`
- `evidence_sufficiency`
- `confidence`
- `unresolved_fields`
- `action_count`
- `no_gain_steps`


## Documentation

A longer project document is available in `docs/CTIAgent.pdf`.
It provides the Introduction, Data Acquisition, Methods, Case Study, Conclusions and Future Work.

## Research and Citation 📚

If you use this project in a paper, report, thesis, or study, please cite this repository:

```bibtex
@misc{CTIAgent2026,
  author  = {Yuyan Chen},
  title   = {CTIAgent: Active Information Acquisition for Clinical Trial Intelligence},
  year    = {2026},
  url     = {\url{https://github.com/Yukyin/CTIAgent}}
}
```

## License 📜

Noncommercial use is governed by `LICENSE` (PolyForm Noncommercial 1.0.0).  
Commercial use requires a separate agreement — see `COMMERCIAL_LICENSE.md`.

📨 Commercial inquiries: yolandachen0313@gmail.com






