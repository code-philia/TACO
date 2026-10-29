#!/usr/bin/env python3
"""Build a distillation prompt from a minimal seed prompt plus accepted experience cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt from seed and cards.")
    parser.add_argument("--seed-prompt", type=Path, required=True)
    parser.add_argument("--cards", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.seed_prompt.read_text(encoding="utf-8").strip()
    cards = []
    for path in args.cards:
        cards.extend(json.loads(path.read_text(encoding="utf-8")))
    lines = [seed, "", "Learned general experience:"]
    for index, card in enumerate(cards, 1):
        title = str(card.get("title") or f"Card {index}").strip()
        instruction = str(card.get("instruction") or "").strip()
        if instruction:
            lines.append(f"{index}. {title}: {instruction}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote prompt to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
