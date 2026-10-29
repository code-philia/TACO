# Repo Evolution Knowledge

Repo Evolution Knowledge (REK) is the cleaned, paper-release version of the
research prototype for repository requirement knowledge construction, retrieval,
training-time knowledge injection, and localization evaluation.

The repository exposes the original research logic through a stable Python
package and CLI while excluding local experiment logs, generated knowledge bases,
private configuration, and machine-specific paths.

## What Is Included

- `src/rek/cli.py`: top-level CLI entrypoint.
- `src/rek/commands/`: Typer command groups for `build`, `train`, `eval`, `kb`, and `inspect`.
- `src/rek/services/`: orchestration layer used by both the CLI and Python API.
- `src/rek/models/`: dataclass request and response objects.
- `src/rek/api/`: Python integration functions.
- `src/rek/parity_core/`: self-contained parity copy of the research core.
- `experiments/prompt_optimization/`: historical-PR prompt optimization, distillation, two-index retrieval, and scoring scripts.
- `docs/CORE_LOGIC.md`: detailed map from paper concepts to implementation files.
- `examples/agent_hint_demo.py`: minimal external-agent style usage example.
- `config.example.toml`: safe configuration template.

## Core Logic

REK has three main flows.

1. Knowledge construction:
   scan a target repository, build a repository knowledge graph, and upgrade it
   into a two-index file/function structure.

2. Retrieval and hinting:
   query the knowledge base with graph-only, hybrid lexical/vector, or parity
   Track A + Track B retrieval. Parity mode combines graph-agent navigation,
   lightweight vector retrieval, LLM arbitration, and historical commit
   projection.

3. Training and evaluation:
   distill knowledge from historical PR patches, inject distilled knowledge into
   the two-index graph, and evaluate bug localization at file, module, and
   function levels.

## Install

```bash
python3 -m pip install -e .
```

After installation:

```bash
rek --help
```

Without installing, use:

```bash
PYTHONPATH=src python3 -m rek.cli --help
```

## Configuration

Start from the safe template:

```bash
cp config.example.toml YOUR_CONFIG_PATH_HERE
```

Then fill in local paths and credentials through environment variables where
possible:

```toml
[llm]
model = "gpt-4o"
api_key = "${OPENAI_API_KEY}"
base_url = "${OPENAI_API_BASE}"

[paths]
repo_path = "YOUR_REPO_PATH_HERE"
kb_path = "YOUR_KB_PATH_HERE"
elements_path = "YOUR_ELEMENTS_PATH_HERE"

[retrieval]
mode = "hybrid"
```

`YOUR_CONFIG_PATH_HERE` is ignored by git and should not be committed.

## Retrieval Modes

- `graph`: deterministic graph-only retrieval over KB rationales and tags.
- `hybrid`: graph retrieval plus lightweight lexical/vector retrieval with RRF.
- `parity`: closest path to the research prototype, using the Track A graph
  agent, Track B vector retrieval, LLM arbitration, and historical projection.

`graph` and `hybrid` are useful for local smoke tests because they can run
without the full LangGraph agent loop. `parity` requires LangChain/LangGraph and
an OpenAI-compatible chat endpoint.

## Common Commands

Inspect configured paths:

```bash
PYTHONPATH=src python3 -m rek.cli inspect paths \
  --repo-path YOUR_REPO_PATH_HERE \
  --kb YOUR_KB_PATH_HERE \
  --json
```

Show KB statistics:

```bash
PYTHONPATH=src python3 -m rek.cli kb stats \
  --config YOUR_CONFIG_PATH_HERE \
  --json
```

Query a KB:

```bash
PYTHONPATH=src python3 -m rek.cli kb query \
  --config YOUR_CONFIG_PATH_HERE \
  --question "Where is the failing behavior implemented?" \
  --json
```

Generate an agent-facing hint:

```bash
PYTHONPATH=src python3 -m rek.cli kb hint \
  --config YOUR_CONFIG_PATH_HERE \
  --issue "Issue statement or bug report text" \
  --json
```

Run localization evaluation:

```bash
PYTHONPATH=src python3 -m rek.cli eval localization \
  --config YOUR_CONFIG_PATH_HERE \
  --test-data YOUR_EVAL_DATA_PATH_HERE
```

## Research Experiment Code

The historical PR prompt optimization workflow is included under
`experiments/prompt_optimization/`. It covers:

- seed and candidate distillation prompts
- historical PR train/dev/held-out split construction
- prompt-based PR knowledge distillation into two-index KBs
- LightVector two-index retrieval
- file/module/function retrieval scoring
- leakage checks and closed-loop prompt revision

Start with:

```bash
cat docs/CORE_LOGIC.md
cat experiments/prompt_optimization/README.md
```

## Smoke Test

Run the self-contained smoke suite:

```bash
PYTHONPATH=src python3 tests/smoke_test.py
```

The smoke test creates a temporary tiny git repository and KB. It covers AST
scanning, two-index upgrade, KB injection/evolution, graph and hybrid retrieval,
patch-derived gold extraction, distillation JSON parsing, historical PR split
generation, two-index retrieval scoring, prompt leakage checking, report
generation, and the CLI entrypoint. It does not call external LLM APIs.

## Data Policy

This open-source tree intentionally does not include:

- private API keys or GitHub tokens
- `YOUR_CONFIG_PATH_HERE`
- generated KB artifacts
- vector caches
- experiment result directories and local data splits
- training and evaluation logs
- large SWE-bench or repository datasets

Place datasets and generated artifacts outside the repository or under ignored
local directories.

## Runtime Output

Training and injection logs default to the `TRAINER_LOG_ROOT` environment
variable. Set it explicitly before running training or injection jobs:

```bash
export TRAINER_LOG_ROOT=YOUR_LOG_DIR_HERE
```

The automatic build pipeline also respects:

```bash
export REK_BASE_REPOS=YOUR_REPO_CHECKOUTS_DIR_HERE
export REK_TRAIN_DATA_DIR=YOUR_TRAIN_DATA_DIR_HERE
```

## Notes For Paper Artifact Review

The code in `src/rek/parity_core/` is intentionally close to the original
research implementation, while the CLI, service, model, and API layers provide
cleaner interfaces for reproducibility and external use. See `MIGRATION_MAP.md`
for the source-to-target mapping.
