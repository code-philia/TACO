# Repo2Doc Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**🔄 [Chinese documentation](./readme.zh.md)**

An agent-driven code repository requirements document generator based on LangGraph. Unlike traditional incremental approaches, this tool uses an **active exploration** strategy.

## ✨ Features

- 🤖 **Agent-Driven Exploration** - Autonomously explores codebase using tools
- 🔄 **Iterative Refinement** - Self-assesses document completeness and iterates
- 🛠️ **Rich Tool Set** - File reading, code analysis, code search capabilities
- 📊 **Confidence Scoring** - Provides confidence score for generated documents
- 📈 **Detailed Statistics** - Tracks token usage and tool call metrics

## 🏗️ How It Works

```
┌─────────────────────┐
│   Initialize        │  Collect README, directory tree, config files
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Generate Doc      │◄──────┐  Generate/update requirements document
└──────────┬──────────┘       │
           │                  │
           ▼                  │
┌─────────────────────┐       │
│ Check Completeness  │       │  LLM evaluates document quality
└──────────┬──────────┘       │
           │                  │
       Complete?              │
      /        \              │
    Yes         No            │
     │           │            │
     ▼           ▼            │
┌─────────┐  ┌─────────────┐  │
│  Save   │  │Execute Tools│──┘  Call tools to gather more info
└─────────┘  └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
cd repo2docAgent
uv sync
```

### Configuration

1. Create `.env` file:
```bash
cp .env.example .env
```

2. Set your API key:
```bash
OPENAI_API_KEY="your-api-key-here"
```

3. (Optional) Customize `config.yaml`:
```yaml
agent:
  max_iterations: 10
  max_tool_calls_per_iteration: 5

llm:
  model: "gpt-4o"
  temperature: 0.3
```

### Usage

```bash
# Basic usage
uv run python main.py YOUR_REPO_PATH_HERE

# Verbose mode (shows all LLM calls and tool executions)
uv run python main.py YOUR_REPO_PATH_HERE -v

# Limit iterations
uv run python main.py YOUR_REPO_PATH_HERE -m 5
```

## 🛠️ Available Tools

The agent can use the following tools to explore the codebase:

| Tool | Description |
|------|-------------|
| `get_file_content` | Read file contents |
| `get_directory_tree` | Get directory structure |
| `list_files_by_extension` | List files by extension |
| `get_file_outline` | Get file outline (classes, functions) |
| `get_function_info` | Get function details |
| `get_class_info` | Get class details |
| `search_code` | Search code across repository |
| `search_imports` | Search import statements |

## 📁 Output Structure

```
repo2docAgent-output/
├── requirements.md              # Final requirements document
├── {timestamp}_requirements.md  # Timestamped backup
├── {timestamp}_report.md        # Exploration report
├── {timestamp}_stats.json       # Token usage & tool statistics
└── intermediate/                # Document versions (if enabled)
    ├── version_1.md
    ├── version_2.md
    └── ...
```

## 📂 Project Structure

```
repo2docAgent/
├── main.py              # CLI entry point
├── agent_workflow.py    # LangGraph agent workflow
├── state.py             # State management
├── config_loader.py     # Configuration loader
├── nodes/               # Workflow nodes
│   ├── init_node.py     # Initialization
│   ├── doc_node.py      # Document generation
│   ├── check_node.py    # Completeness check
│   ├── tool_node.py     # Tool execution
│   └── save_node.py     # Output saving
├── tools/               # Agent tools
│   ├── file_tools.py    # File operations
│   ├── code_tools.py    # Code analysis
│   └── search_tools.py  # Code search
└── prompts/             # Prompt templates
    └── agent_prompts.py
```

## 🆚 Comparison: Repo2Doc vs Repo2Doc Agent

| Aspect | Repo2Doc | Repo2Doc Agent |
|--------|----------|----------------|
| **Approach** | Incremental chunking | Agent exploration |
| **File Access** | All files upfront | On-demand via tools |
| **Iterations** | One pass per chunk | Multiple refinement cycles |
| **Flexibility** | Pre-planned | Adaptive |
| **Best For** | Smaller codebases | Complex projects |

## 📄 License

MIT License

---

**Related Project**: [Repo2Doc](../repo2doc) - The incremental chunking variant.
