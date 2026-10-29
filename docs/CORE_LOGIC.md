# Core Logic Map

This document maps the research functions that matter for the paper artifact to
their implementation points in this repository.

## 1. Distillation Prompts

The core training prompt asks an LLM to turn a historical PR issue and patch
into reusable KB entries:

```json
[
  {
    "file": "path/to/file.py",
    "function": "function_or_FILE_SUMMARY",
    "keywords": ["searchable", "technical", "terms"],
    "rationale": "code-grounded reusable knowledge"
  }
]
```

Important prompt assets:

- `experiments/prompt_optimization/seeds/simplified_distillation_prompt.txt`
- `experiments/prompt_optimization/seeds/current_six_step_prompt.txt`
- `experiments/prompt_optimization/prompts/process_seed_prompt.txt`
- `src/rek/parity_core/pr_distiller.py`

The six-step prompt logic is:

1. remove PR noise
2. reconstruct user-visible semantic intent
3. align with production-code diff evidence
4. bridge issue language to changed code logic
5. synthesize searchable keywords
6. emit file-level and function-level knowledge

## 2. Two-Index KB Construction

The two-index KB has:

- file nodes: `file_path`, feature links, role, rationale, keywords
- function nodes: function/method name, type, rationale, keywords
- optional program elements: code locations and snippets in `program_elements.jsonl`

Main code:

- `src/rek/parity_core/scanner.py`: scans Python AST into program elements
- `src/rek/parity_core/repo2doc_bridge.py`: runs repo2doc local phases
- `src/rek/parity_core/repo2doc_local/`: local graph construction workflow
- `src/rek/parity_core/upgrade_kb_two_index.py`: upgrades file-level KB to file/function two-index KB
- `src/rek/services/build_service.py`: CLI-facing construction service

Typical outputs:

- `knowledge_graph_complete_detail.json`
- `knowledge_graph_two_index.json`
- `program_elements.jsonl`
- `scanned_files_stats.jsonl`

## 3. PR Knowledge Distillation

Historical PR distillation has three steps.

First, extract patch-derived historical gold and time mapping:

- `src/rek/parity_core/pr_distiller.py`
- `GitTimeMachine.build_time_mapping`
- `ASTGroundTruthExtractor.extract_3level_gt`

The time mapping connects historical changed files/functions to modern KB files
and available modern functions.

Second, call the distillation prompt:

- `PRKnowledgeDistiller.distill`
- `experiments/prompt_optimization/scripts/distill_manifest_to_kb.py`

Third, normalize JSON output and preserve only items with valid `file` and
`rationale` fields.

## 4. KB Evolution By Injection

Distilled PR knowledge evolves the KB by appending rationale and keywords at the
file or function level.

Main code:

- `src/rek/parity_core/graph_db.py`
- `GraphDatabase.update_node_knowledge`
- `src/rek/parity_core/workflow.py`
- `EvolutionGraph.run_fast_injection`
- `src/rek/parity_core/mutation_tools.py`
- `inject_dual_index_knowledge`
- `src/rek/parity_core/skills/graph-injection/SKILL.md`

Training orchestration:

- `src/rek/parity_core/trainer.py`
- `run_training`
- `src/rek/services/train_service.py`

The training loop loads historical PR samples, distills each patch into
knowledge items, writes an epoch KB copy, and injects distilled knowledge into
the two-index graph.

## 5. Two-Index Retrieval

There are three retrieval surfaces.

### Graph Mode

Deterministic graph retrieval over file/function rationales, keywords, and graph
ancestry.

- `src/rek/services/graph_retriever.py`
- `rek kb query --config ...` with `[retrieval].mode = "graph"`

### Hybrid Mode

Graph retrieval plus `LightVectorStore` lexical/vector retrieval, fused with
reciprocal rank fusion.

- `src/rek/services/hybrid_retriever.py`
- `src/rek/services/vector_retriever.py`
- `src/rek/parity_core/light_vector.py`
- `rek kb query --config ...` with `[retrieval].mode = "hybrid"`

### Parity Mode

The research retrieval workflow:

1. Track A: graph agent navigation using `get_root_categories`,
   `get_l2_features`, `get_files_in_feature`, `get_functions_in_file`, and
   `search_by_tags`.
2. Track B: `LightVectorStore.search` over file/function rationale text.
3. Synergy: compare Track A and Track B overlap and disagreements.
4. Arbiter: synthesize final candidates.
5. Historical projection: map modern candidates back to the target base commit.

Main code:

- `src/rek/parity_core/workflow.py`
- `EvolutionGraph.run_inference_for_single_sample`
- `src/rek/parity_core/inference_tools.py`
- `src/rek/parity_core/git_time_travel.py`
- `src/rek/parity_core/skills/retrieval-navigation/SKILL.md`

## 6. Prompt Optimization Experiment

The prompt optimization experiment evaluates whether a distillation prompt
generates KB entries that improve later retrieval.

Main loop:

1. `build_historical_splits.py`: create train/dev/heldout manifests with
   patch-derived file/module/function gold.
2. `distill_manifest_to_kb.py`: generate per-repo two-index KBs from train PRs.
3. `run_lightvector_retrieval.py`: retrieve over those two-index KBs for later
   PR queries.
4. `score_retrieval.py`: score function, module, and file hits.
5. `optimize_prompt_from_sample_scores.py`: generate a revised generic prompt
   from abstracted score feedback.
6. `check_prompt_leakage.py`: reject prompts that copy sample-specific terms.

Experiment code lives under:

- `experiments/prompt_optimization/`

## 7. Evaluation

Localization evaluation compares predicted candidates with patch-derived or
official annotations at three levels:

- file
- module/class/global scope
- function/method

Main code:

- `src/rek/parity_core/eval_localization.py`
- `src/rek/services/eval_service.py`
- `rek eval localization`

## 8. Data Boundaries

The open-source repository includes method code and prompt assets. It excludes:

- historical PR JSONL datasets
- repository checkouts
- generated KBs
- prompt optimization results
- private API keys and endpoint configs

Use `REK_TRAIN_DATA_DIR`, `REK_BASE_REPOS`, `OPENAI_API_KEY`, and
`OPENAI_BASE_URL` to point the code at local resources.
