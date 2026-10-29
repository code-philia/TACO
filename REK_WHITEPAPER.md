# Repo Evolution Knowledge Whitepaper

## Purpose

Repo Evolution Knowledge (REK) is a research artifact for building and using
repository-level requirement knowledge. It turns repository history, source-code
structure, and patch-derived experience into a two-index knowledge graph that
can assist bug localization and agent navigation.

This paper-release repository keeps the research logic available while exposing
it through reproducible command-line and Python interfaces.

## Architecture

REK is organized into five layers:

- `src/rek/cli.py`: top-level CLI entrypoint.
- `src/rek/commands/`: CLI argument parsing and command groups.
- `src/rek/services/`: orchestration layer for build, train, eval, inspect, and KB access.
- `src/rek/models/`: request and response dataclasses.
- `src/rek/parity_core/`: self-contained copy of the research core.

The `parity_core` package is intentionally close to the internal prototype. The
outer CLI/service/API layers provide a cleaner boundary for artifact users.

## Knowledge Representation

The main artifact is a two-index knowledge graph:

- file-level nodes contain repository role, feature links, rationale text, and tags
- function-level nodes contain signatures, local rationale, and function tags
- optional `program_elements.jsonl` records code elements, locations, and snippets

The graph format supports both deterministic inspection and agent-guided
navigation.

## Core Flows

### Construction

The construction pipeline scans Python source files, builds or imports a
repository knowledge graph, and upgrades it to the two-index format.

Important modules:

- `src/rek/parity_core/scanner.py`
- `src/rek/parity_core/repo2doc_bridge.py`
- `src/rek/parity_core/upgrade_kb_two_index.py`
- `src/rek/services/build_service.py`

### Retrieval

REK supports three retrieval modes:

- `graph`: deterministic graph retrieval over rationales and tags
- `hybrid`: graph retrieval plus lightweight lexical/vector retrieval with reciprocal rank fusion
- `parity`: the closest reproduction of the research workflow

Parity retrieval has four steps:

1. Track A: an agent navigates the knowledge graph with graph tools.
2. Track B: lightweight vector retrieval searches file and function rationales.
3. Synergy and arbitration: an LLM compares Track A and Track B evidence.
4. Historical projection: modern candidates are mapped back to the target base commit.

Important modules:

- `src/rek/parity_core/workflow.py`
- `src/rek/parity_core/graph_db.py`
- `src/rek/parity_core/light_vector.py`
- `src/rek/services/kb_service.py`

### Training

Training distills historical PR patches into reusable knowledge and injects it
into the two-index graph. The intent is to improve future localization by
recording recurring implementation signals at file and function granularity.

Important modules:

- `src/rek/parity_core/pr_distiller.py`
- `src/rek/parity_core/trainer.py`
- `src/rek/parity_core/mutation_tools.py`
- `src/rek/services/train_service.py`

### Evaluation

Localization evaluation compares predicted files, modules, and functions against
ground truth extracted from patches or supplied annotation files.

Important modules:

- `src/rek/parity_core/eval_localization.py`
- `src/rek/parity_core/git_time_travel.py`
- `src/rek/services/eval_service.py`

## CLI Surface

Top-level commands:

- `rek build run`
- `rek train run`
- `rek eval localization`
- `rek kb stats`
- `rek kb search`
- `rek kb query`
- `rek kb retrieve`
- `rek kb hint`
- `rek inspect paths`

The same functionality can be accessed from Python through `src/rek/api/`.

## Configuration And Reproducibility

Use `config.example.toml` as the template. Local credentials and absolute data
paths should be stored in ignored local config files or environment variables.

The repository intentionally does not ship generated KBs, SWE-bench datasets,
private repository checkouts, logs, caches, or API tokens. Artifact users should
place those resources outside the source tree or in ignored local directories.

Runtime logs should be configured with placeholder-backed environment variables.
Set the following variables to local paths before running experiments:

- `TRAINER_LOG_ROOT`
- `REK_BASE_REPOS`
- `REK_TRAIN_DATA_DIR`

## Relation To The Development Trees

This open-source tree was assembled from an original research prototype and an
intermediate CLI refactor. See `MIGRATION_MAP.md` for the module-level mapping.
The result is a single self-contained repository suitable for paper artifact
release.
