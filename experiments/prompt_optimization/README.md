# Historical PR Prompt Optimization

This experiment studies how to optimize the knowledge-distillation prompt using
only historical PR feedback. The generated knowledge is injected into two-index
KB files and then evaluated with two-index retrieval.

## Data Policy

SWE-bench Lite or Verified must not be used for prompt training, dev selection,
error attribution, prompt rewriting, or candidate ranking. SWE-bench is reserved
for external evaluation after a prompt is frozen.

The open-source tree includes code and prompt assets only. It does not include
historical PR JSONL files, generated KBs, results, logs, or private secrets.

## Main Components

- `seeds/`: human-written seed prompts.
- `prompts/`: selected prompt candidates retained as research artifacts.
- `scripts/build_historical_splits.py`: builds train/dev/held-out manifests with patch-derived file/module/function gold.
- `scripts/distill_manifest_to_kb.py`: runs a prompt over historical PRs and writes per-repo two-index KB JSON.
- `scripts/run_lightvector_retrieval.py`: runs two-index LightVector retrieval over query manifests.
- `scripts/score_retrieval.py`: scores retrieved file/function candidates against patch-derived gold.
- `scripts/run_closed_loop_iteration.py`: distill, retrieve, score, and generate the next prompt candidate.
- `scripts/run_safe_atomic_loop.py`: conservative multi-epoch prompt-edit loop.

## Environment

Set local data locations outside the repository:

```bash
export REK_TRAIN_DATA_DIR=YOUR_TRAIN_DATA_DIR_HERE
export REK_BASE_REPOS=YOUR_REPO_CHECKOUTS_DIR_HERE
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=YOUR_OPENAI_COMPATIBLE_BASE_URL_HERE
```

## Build Historical Splits

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/build_historical_splits.py \
  --train-data-dir "$REK_TRAIN_DATA_DIR" \
  --base-repos "$REK_BASE_REPOS" \
  --output-dir experiments/prompt_optimization/data_splits/sample \
  --heldout-repos sphinx sympy \
  --dev-ratio 0.2 \
  --min-history 50
```

Each row contains:

- `query`: PR title/problem text used as a future issue query
- `patch`: historical patch text
- `base_commit`: commit used for historical gold extraction
- `gold.files`, `gold.modules`, `gold.functions`: patch-derived labels

## Distill PR Knowledge Into Experiment KBs

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/distill_manifest_to_kb.py \
  --manifest experiments/prompt_optimization/data_splits/sample/train.jsonl \
  --prompt experiments/prompt_optimization/seeds/simplified_distillation_prompt.txt \
  --output-dir experiments/prompt_optimization/results/sample_run \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL"
```

This writes per-repo KB files under:

```text
<output-dir>/kb_repos/<repo>/repo2docAgent-output/knowledge_graph_two_index.json
```

The generated KB schema is compatible with `rek.parity_core.graph_db.GraphDatabase`
and `rek.parity_core.light_vector.LightVectorStore`.

## Run Two-Index Retrieval

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/run_lightvector_retrieval.py \
  --manifest experiments/prompt_optimization/data_splits/sample/dev.jsonl \
  --repo-root experiments/prompt_optimization/results/sample_run \
  --output experiments/prompt_optimization/results/sample_run/dev_retrieval.jsonl \
  --output-dir-name repo2docAgent-output \
  --kb-name knowledge_graph_two_index.json \
  --top-k 20
```

## Score Retrieval

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/score_retrieval.py \
  --results experiments/prompt_optimization/results/sample_run/dev_retrieval.jsonl \
  --output-summary experiments/prompt_optimization/results/sample_run/dev_score_summary.json \
  --output-details experiments/prompt_optimization/results/sample_run/dev_score_details.jsonl \
  --top-k 20
```

The score prefers exact function hits, then module hits, then file hits.

## Closed-Loop Prompt Optimization

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/run_closed_loop_iteration.py \
  --iteration-name iter_000 \
  --prompt experiments/prompt_optimization/seeds/simplified_distillation_prompt.txt \
  --splits-dir experiments/prompt_optimization/data_splits/sample \
  --work-dir experiments/prompt_optimization/results/closed_loop \
  --prompts-dir experiments/prompt_optimization/prompts \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL"
```

The loop:

1. distills train PRs into per-repo two-index KBs
2. retrieves for train/dev queries
3. scores file/module/function hits
4. builds abstracted feedback
5. asks an LLM to propose a generic prompt revision
6. checks leakage against train/dev manifests

## Prompt Leakage Check

```bash
PYTHONPATH=src python3 experiments/prompt_optimization/scripts/check_prompt_leakage.py \
  --prompt experiments/prompt_optimization/prompts/candidate.txt \
  --manifest experiments/prompt_optimization/data_splits/sample/train.jsonl \
  --manifest experiments/prompt_optimization/data_splits/sample/dev.jsonl \
  --output experiments/prompt_optimization/results/leakage_report.json
```

A candidate prompt should not contain sample-specific repository names, file
paths, function names, issue identifiers, or concrete APIs from the train/dev
data.
