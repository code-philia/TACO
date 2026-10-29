from __future__ import annotations

import argparse

from rek.parity_core.github_pr_collector import collect_pr_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity launcher for GitHub PR data collection.")
    parser.add_argument("--repo", required=True, help="Repository name.")
    parser.add_argument("--owner", required=True, help="Repository owner.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    args = parser.parse_args()
    collect_pr_data(repo_name=args.repo, owner=args.owner, output_path=args.output)


if __name__ == "__main__":
    main()
