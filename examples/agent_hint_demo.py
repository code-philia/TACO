from __future__ import annotations

import argparse
import json

from rek.api import retrieve_issue_hint
from rek.config import apply_runtime_environment, load_config, resolve_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal external-agent style hint retrieval demo.")
    parser.add_argument("--config", help="Path to TOML config file.", default=None)
    parser.add_argument("--kb", help="Knowledge base path override.", default=None)
    parser.add_argument("--repo-path", help="Repository path override.", default=None)
    parser.add_argument("--elements-path", help="Program elements path override.", default=None)
    parser.add_argument("--issue", required=True, help="Issue statement or bug description.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of retrieval candidates.")
    parser.add_argument(
        "--max-hint-candidates",
        type=int,
        default=5,
        help="Maximum number of candidates rendered into the hint.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON response.")
    args = parser.parse_args()

    runtime_config = load_config(args.config)
    apply_runtime_environment(runtime_config)

    kb_path = resolve_value(args.kb, runtime_config.paths.kb_path)
    repo_path = resolve_value(args.repo_path, runtime_config.paths.repo_path)
    elements_path = resolve_value(args.elements_path, runtime_config.paths.elements_path)

    if not kb_path:
        raise SystemExit("A KB path is required via --kb or --config.")

    result = retrieve_issue_hint(
        kb_path=kb_path,
        issue=args.issue,
        repo_path=repo_path,
        elements_path=elements_path,
        top_k=args.top_k,
        max_hint_candidates=args.max_hint_candidates,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result.get("hint", ""))


if __name__ == "__main__":
    main()
