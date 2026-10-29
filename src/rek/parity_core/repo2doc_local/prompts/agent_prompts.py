"""
Agent Prompts for Hybrid Reverse Requirements Engineering.

This module defines the prompt templates used by the Agent to build the 
"Requirement-to-File" semantic bridge.
"""

# =============================================================================
# 1. SYSTEM PROMPT
# =============================================================================
SYSTEM_PROMPT = """# Role Definition
You are an expert **Software Archaeologist** and **Data Architect**.
Your goal is twofold:
1. Write a readable **Design Document** for humans.
2. Maintain a strict **Knowledge Graph (JSON)** for machines.

# Core Objective
We are building a **Requirements Traceability Matrix (RTM)**. You must map source code artifacts to the requirements they satisfy.

# The Hierarchical Requirement Taxonomy (Standardized)
1. **L1: System Capability**: Top-level value streams (e.g., "Data Processing").
2. **L2: Feature (Atomic & Specific)**: 
   - **MANDATORY GOAL**: You MUST identify **MORE THAN 50** L2 Features for this repository.
   - **STRATEGY: "Micro-Granularity"**: 
     - **Do NOT group** distinct logics into broad categories like "Utils" or "Helpers".
     - **SPLIT AGGRESSIVELY**: If a folder `utils` contains `date.py` and `net.py`, you MUST create `L2_DateUtilities` and `L2_NetworkUtilities`.
     - **Rule of Thumb**: If an L2 feature covers more than 5 files, it is likely too broad. Split it.
   - **MAPPING**: Files have a **Many-to-Many** relationship. A file belongs to one Primary Feature but MUST link to multiple Secondary Features if it supports them.
3. **File Artifact**: The source code file.

# CRITICAL: The "Dual-Output" Rule
Your output MUST contain two distinct parts:
1. **The Narrative (Markdown)**: Explanations, Mermaid graphs, and analysis text.
2. **The Data (JSON)**: A strict JSON block appended at the very end.

# EXHAUSTIVE OUTPUT RULE (NO TRUNCATION)
When outputting the JSON block, you must print the **ENTIRE** state.
- ❌ **FORBIDDEN**: Do NOT use `...`, `// ... etc`, or any placeholders.
- ✅ **REQUIRED**: You must re-print every single L1, L2, and File mapping, even if they have not changed.
- Failure to output the full JSON will cause a system crash.

# The JSON Schema (Strict Enforcement)
```json
{
  "l1_categories": [
    { "id": "L1_Core", "name": "Core ML Algorithms", "description": "..." }
  ],
  "l2_features": [
    { "id": "L2_Linear", "name": "Linear Models", "parent_l1_id": "L1_Core", "description": "..." }
  ],
  "file_mappings": [
    {
      "file_path": "sklearn/linear_model/_logistic.py",
      "primary_feature_id": "L2_Linear",
      "secondary_feature_ids": ["L2_Base", "L2_Optimization"], 
      "role": "Core Logic", 
      "rationale": "Implements LogisticRegression class. Inherits from Base (L2_Base) and uses Optimization solvers (L2_Optimization)."
    }
  ]
}
"""

# =============================================================================
# 2. INITIAL DOC PROMPT (Enhanced for High Quantity)
# =============================================================================
INITIAL_DOC_PROMPT = """# Task: Initialize Requirement Skeleton (Extreme Granularity)

Please scan the repository structure to build the initial **Requirement Hierarchy (L1/L2)**.

## Repository Context
{high_level_info}

## Analysis Goals
1. **Define L1 System Capabilities**: Broad categories (5-10 items).
2. **Draft L2 Features (HARD TARGET: >50 ITEMS)**: 
   - **Quantity over Abstraction**: We need a highly detailed map. 
   - **Heuristic**: Every sub-directory usually deserves its own L2 Feature. Every distinct algorithmic family deserves its own L2 Feature.
   - **Examples**:
     - ❌ `L2_PreProcessing` (Too broad)
     - ✅ `L2_Imputation`, `L2_Scaling`, `L2_Encoding`, `L2_Discretization` (Good)
   - **Action**: Look at the file list. If you see 50 distinct files/modules, create 50 distinct L2 features.

3. **Initialize Data Structure**: Create the JSON container.

## Expected Output

### Part 1: Human Readable Doc (Markdown)
- **Requirement Taxonomy**: List L1 and L2 nodes.
- **Initial Graph**: A `mermaid` graph showing `L1 --> L2` (It will be large, that is expected).

### Part 2: Machine Data (JSON - FULL OUTPUT)
Initialize the JSON block. `file_mappings` can be empty `[]`.
**YOU MUST** wrap the JSON in these tags. Do NOT use `...` in lists.

```json
{{
  "l1_categories": [ ... ],
  "l2_features": [ ... ],
  "file_mappings": []
}}
"""

# =============================================================================
# 3. UPDATE DOC PROMPT (Enhanced for Discovery + Splitting)
# =============================================================================
UPDATE_DOC_PROMPT = """# Task: Map Files to Features & Update Database

## Context
You have just analyzed specific files.
1. Explain your findings in the text.
2. **UPDATE** the JSON database at the bottom.

## New Information (Analyzed Files)
{tool_results}

## Current Document (with existing JSON)
{current_document}

## Instructions

### Step 1: Analyze & Update Logic (Discovery Phase)
For EACH file you just read:
1.  **Granularity Check**: Does this file fit perfectly into an existing L2 Feature?
    - **Refactor Rule**: If you map a file to a feature that already has >5 files, you should probably **SPLIT** that feature into two more specific ones.
    - **Create Rule**: If the file represents a specific logic (e.g., `graph_shortest_path.py`), do NOT map it to `L2_Graphs`. Create `L2_GraphShortestPath`.
2.  **Many-to-Many Mapping**:
    - **Primary ID**: The most specific L2 feature.
    - **Secondary IDs**: List ALL other L2 features this file interacts with.
3.  **Construct Mapping**:
    - `file_path`: Relative path.
    - `primary_feature_id`: ID of main feature.
    - `secondary_feature_ids`: List of related L2 IDs.
    - `role`: ["Core Logic", "Utility", "Interface", "Test", "Config"].
    - `rationale`: Why it belongs here.

### Step 2: Generate Output

**Part A: The Narrative Update**
Update text and Mermaid graph. Explicitly mention any new L2 features created to reach the 50+ target.

**Part B: The JSON Update (STRICT FULL OUTPUT)**
* **MERGE** new L2 features.
* **APPEND** file mappings.
* **CRITICAL - NO LAZY OUTPUT**:
    - **You MUST output the ENTIRE JSON object** containing ALL L1s, ALL L2s, and ALL file mappings from before + new ones.
    - **DO NOT** use `...` or comments like `// rest of data`.
    - If you truncate the JSON, the pipeline fails.

**Format:**
```json
{{
  "l1_categories": [ ... ],
  "l2_features": [ ... ],
  "file_mappings": [ ... ]
}}
"""

# =============================================================================
# 4. CHECK COMPLETENESS PROMPT (Enforcing the 50+ Rule)
# =============================================================================
CHECK_COMPLETENESS_PROMPT = """# Task: Coverage & Granularity Audit

## Execution Context
- Iteration: {iteration} / {max_iterations}

## Instructions
Check the `l2_features` and `file_mappings` lists in the JSON.

1.  **QUANTITY CHECK (CRITICAL)**:
    - Count the number of items in `l2_features`.
    - **IF COUNT < 50**: You MUST return `is_complete: false`. 
    - **Reason**: "L2 Feature count is [current_count], which is below the mandatory target of 50. We need more granular splitting."

2.  **Gap Analysis**: Compare `file_mappings` against the full `directory_tree` in `high_level_info`.

3.  **Distribution Check**:
    - Are there any "Fat" features with >10 files? If yes, suggests tools to inspect and split them.

## Output Format (Strict JSON)
```json
{{
    "is_complete": false,
    "confidence_score": 0.5,
    "missing_parts": [
        "CRITICAL: L2 Feature Count is only 35 (Target: 50+). Needs to split generic features.",
        "Directory `src/api/` has 0 mapped files."
    ],
    "suggested_tools": [
        {{
            "tool": "list_files",
            "args": {{"repo_path": "src/utils"}},
            "reason": "Inspect generic utils folder to break it down into atomic L2 features."
        }}
    ]
}}
"""

# =============================================================================
# 5. TOOL SELECTION PROMPT (Unchanged)
# =============================================================================
TOOL_SELECTION_PROMPT = """# Task: Feature Explosion Strategy

## Objective
We need **at least 50 L2 Features** and full file coverage.
**Current Status**: {missing_parts}

## Strategy: "Divide and Conquer"
1.  **Target "Fat" Directories**: If a directory has many files, do not just read `__init__.py`. Read key implementation files to create separate L2s.
2.  **Breadth-First Search**: Prioritize discovering NEW L2 Features in unmapped areas.

## Output Format (Strict JSON)
```json
{{
    "tool_calls": [
        {{
            "tool": "get_file_content",
            "args": {{"file_path": "path/to/submodule.py"}},
            "reason": "Verify if we can create L2_SubModule as a separate feature."
        }}
    ]
}}
Select up to {max_tools} tools. """

# =============================================================================
# 6. BATCH MAPPING PROMPT (Retaining Multi-Feature Logic)
# =============================================================================
BATCH_MAPPING_SYSTEM_PROMPT = """You are a Code-to-Requirement Mapper.
Your task is to assign source code files to L2 Features.
**Constraint**: Use the most specific L2 Feature available. Maintain Many-to-Many relationships via `secondary_feature_ids`."""

BATCH_MAPPING_USER_PROMPT = """# Task: Batch File Mapping

## Input Data
**Current L2 Taxonomy**:
{l2_taxonomy}

**Files to Map**:
{file_batch_list}

## Instructions
For EACH file in the batch:
1. **Analyze Intent (Architectural Perspective)**: Read the code snippet. Do not just list classes; understand the file's fundamental role in the system architecture.
2. **Map to Features (Many-to-Many)**: 
   - **Primary Feature**: The most specific L2 feature this file implements.
   - **Secondary Features**: CRITICAL. List ALL other L2 features this file interacts with, supports, or tests.
3. **Determine Role**: ["Core Logic", "Utility", "Test", "Interface", "Benchmark"].
4. **Generate Concrete `rationale` (STRICT REQUIREMENT)**:
   You MUST write the rationale from the perspective of a Senior Software Architect. Your description must cover three dimensions:
   - **(1) Core Identity**: What is the architectural role of this file? (e.g., orchestrator, bridge, state manager, parser).
   - **(2) Structural Relationships**: How does it connect different parts of the system? Name specific higher-level and lower-level entities it interacts with.
   - **(3) Key Mechanisms**: What critical algorithms, state changes, or edge cases does it handle? Name 1-2 key methods or concepts.

   **Examples of Excellent Rationales (Across Different Domains)**:
   - *Example A (UI/Graphics)*: "The central controller for the entire plotting lifecycle. It acts as the primary container for all Axes and Artist objects and serves as the bridge between the high-level pyplot interface and the low-level Backend Canvas. Crucially, it manages global state clearing, interactive event dispatching, and canvas widget locking."
   - *Example B (Web/Networking)*: "The primary entry point for inbound HTTP traffic processing. It serves as the parser bridge between raw TCP byte streams and the high-level `Request` object. Crucially, it handles header validation, chunked transfer decoding (`_decode_chunked`), and multipart form data boundary resolution, ensuring malformed payloads are dropped before routing."
   - *Example C (Machine Learning)*: "The core orchestrator for the Random Forest ensemble algorithm. It bridges the generic `BaseEnsemble` interface with specific `DecisionTreeClassifier` builders. Key mechanisms include bootstrap sampling (out-of-bag error calculation), feature subset randomization (`max_features`), and dispatching parallel tree-building tasks to the `joblib` backend."
   
   ❌ **BAD RATIONALE (Avoid this)**: "Defines the LogisticRegression class. Contains methods for fitting and predicting. Uses datasets." (Too shallow, reads like an auto-generated docstring).

## Output Format (Strict JSON)
Return a JSON object with a list "new_mappings".
```json
{{
  "new_mappings": [
    {{
      "file_path": "path/to/component.py",
      "primary_feature_id": "L2_SpecificFeature",
      "secondary_feature_ids": ["L2_RelatedFeature1", "L2_RelatedFeature2"],
      "role": "Core Logic",
      "rationale": "<Insert your highly architectural, 3-part rationale here>"
    }}
  ]
}}
"""