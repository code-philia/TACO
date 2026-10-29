#!/usr/bin/env python3
"""Apply one atomic prompt edit to produce one prompt variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROCESS_HEADER = "Internal reasoning process to follow silently:"
PROCESS_END = "End internal reasoning process."
INSERT_AFTER = "Extract reusable knowledge that may help locate relevant code for future similar issues."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply one atomic prompt edit.")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--edit-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def load_edits(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("edits") or raw.get("prompt_edits") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def apply_edit(prompt_text: str, edit: dict[str, Any]) -> tuple[str, str]:
    edit_type = str(edit.get("edit_type") or "").strip()
    target = str(edit.get("target") or "").strip()
    text = str(edit.get("text") or "").strip()
    if not text:
        raise ValueError("Edit text is empty")

    if edit_type == "append_bullet":
        bullet = text if text.lstrip().startswith(('-', '*')) else f"- {text}"
        if target and target in prompt_text:
            lines = prompt_text.splitlines()
            out = []
            inserted = False
            for line in lines:
                out.append(line)
                if not inserted and target in line:
                    out.append(bullet)
                    inserted = True
            if not inserted:
                out.extend(["", "Learned general experience:", bullet])
            return "\n".join(out).strip() + "\n", "append_bullet"
        if "Learned general experience:" in prompt_text:
            return prompt_text.rstrip() + "\n" + bullet + "\n", "append_bullet"
        return prompt_text.rstrip() + "\n\nLearned general experience:\n" + bullet + "\n", "append_bullet"

    if edit_type == "replace_process":
        process_text = text.strip()
        if not process_text.startswith(PROCESS_HEADER):
            process_text = f"{PROCESS_HEADER}\n{process_text}"
        if PROCESS_END not in process_text:
            process_text = process_text.rstrip() + f"\n{PROCESS_END}"
        block = process_text.strip()

        if PROCESS_HEADER in prompt_text and PROCESS_END in prompt_text:
            start = prompt_text.index(PROCESS_HEADER)
            end = prompt_text.index(PROCESS_END, start) + len(PROCESS_END)
            return (prompt_text[:start].rstrip() + "\n\n" + block + "\n\n" + prompt_text[end:].lstrip()).strip() + "\n", "replace_process"

        if INSERT_AFTER in prompt_text:
            return prompt_text.replace(INSERT_AFTER, INSERT_AFTER + "\n\n" + block, 1).strip() + "\n", "replace_process"

        return prompt_text.rstrip() + "\n\n" + block + "\n", "replace_process"

    if edit_type == "replace_text":
        if not target or target not in prompt_text:
            raise ValueError("replace_text target not found")
        return prompt_text.replace(target, text, 1).strip() + "\n", "replace_text"

    if edit_type == "remove_text":
        remove_text = target or text
        if remove_text not in prompt_text:
            raise ValueError("remove_text target not found")
        return prompt_text.replace(remove_text, "", 1).strip() + "\n", "remove_text"

    raise ValueError(f"Unsupported edit_type: {edit_type}")


def main() -> int:
    args = parse_args()
    edits = load_edits(args.edits)
    if not edits:
        raise SystemExit("No edits found")
    if args.edit_index < 0 or args.edit_index >= len(edits):
        raise SystemExit(f"edit-index {args.edit_index} out of range for {len(edits)} edits")
    edit = edits[args.edit_index]
    prompt_text = args.prompt.read_text(encoding="utf-8")
    new_prompt, action = apply_edit(prompt_text, edit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(new_prompt, encoding="utf-8")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "source_prompt": str(args.prompt),
                "edits": str(args.edits),
                "edit_index": args.edit_index,
                "action": action,
                "edit": edit,
                "output": str(args.output),
                "prompt_chars": len(new_prompt),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote prompt variant to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
