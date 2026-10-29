# Migration Map

This file documents how the paper-release repository was assembled from the two
internal development trees:

- the original research prototype
- the intermediate CLI refactor

The open-source tree is self-contained and does not import either source tree at
runtime.

## Layer Mapping

- Original research logic was copied into `src/rek/parity_core/`.
- CLI-facing orchestration lives in `src/rek/services/` and `src/rek/commands/`.
- Python integration functions live in `src/rek/api/`.
- Request and response schemas live in `src/rek/models/`.

## Source-To-Target Map

- `src/evolution_pipeline/workflow.py`
  -> `src/rek/parity_core/workflow.py`
- `src/evolution_pipeline/pr_distiller.py`
  -> `src/rek/parity_core/pr_distiller.py`
- `src/evolution_pipeline/skill_loader.py`
  -> `src/rek/parity_core/skill_loader.py`
- `src/evolution_pipeline/trainer.py`
  -> `src/rek/parity_core/trainer.py`
- `src/evolution_pipeline/eval_localization.py`
  -> `src/rek/parity_core/eval_localization.py`
- `src/tools/git_time_travel.py`
  -> `src/rek/parity_core/git_time_travel.py`
- `src/tools/graph_db.py`
  -> `src/rek/parity_core/graph_db.py`
- `src/retrieval/light_vector.py`
  -> `src/rek/parity_core/light_vector.py`
- `src/tools/inference_tools.py`
  -> `src/rek/parity_core/inference_tools.py`
- `src/tools/mutation_tools.py`
  -> `src/rek/parity_core/mutation_tools.py`
- `src/tools/__init__.py`
  -> `src/rek/parity_core/tools_registry.py`
- `support/upgrade_kb_two_index.py`
  -> `src/rek/parity_core/upgrade_kb_two_index.py`
- `support/Repo2TutorialAgent`
  -> `src/rek/parity_core/repo2doc_local/`
- `prompt_optimization_experiment/seeds/`
  -> `experiments/prompt_optimization/seeds/`
- `prompt_optimization_experiment/prompts/`
  -> `experiments/prompt_optimization/prompts/`
- `prompt_optimization_experiment/scripts/`
  -> `experiments/prompt_optimization/scripts/`

## CLI-Native Modules

- `src/rek/cli.py`: top-level CLI entrypoint.
- `src/rek/commands/`: command parsing for `build`, `train`, `eval`, `kb`, and `inspect`.
- `src/rek/services/kb_service.py`: graph, hybrid, and parity KB access.
- `src/rek/services/build_service.py`: construction pipeline orchestration.
- `src/rek/services/train_service.py`: training pipeline orchestration.
- `src/rek/services/eval_service.py`: localization evaluation orchestration.
- `src/rek/api/kb.py`: Python-facing API.
- `src/rek/skills/kb_tools.py`: skill/tool wrapper candidates.
- `experiments/prompt_optimization/`: historical PR prompt optimization and two-index retrieval evaluation.
- `docs/CORE_LOGIC.md`: paper-level logic map for prompts, construction, distillation, KB evolution, and retrieval.

## Open-Source Cleanup

The paper-release tree excludes:

- local configs with credentials
- generated KB files and vector caches
- training and evaluation logs
- prompt optimization experiment outputs
- prompt optimization data splits, logs, and generated KB results
- package build metadata such as egg-info
- large local datasets and repository checkouts

Machine-specific default paths were replaced with environment-variable-driven
or current-working-directory defaults.
