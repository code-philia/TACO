"""
Local copy adapted from `Repo_Reqirement_Knowledge/src/evolution_pipeline/skill_loader.py`.

This loader keeps the original skill file behavior while resolving paths inside
the CLI project.
"""

from __future__ import annotations

import re
from pathlib import Path


class SkillRegistry:
    def __init__(self, skills_root: str | None = None):
        if skills_root is None:
            self.root = Path(__file__).resolve().parent / "skills"
        else:
            self.root = Path(skills_root)

    def load_skill_content(self, skill_name: str) -> str:
        """
        Load the skill body, remove frontmatter, and expand `{{LOAD:...}}` references.
        """
        skill_dir = self.root / skill_name
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            return f"[System Error: Skill '{skill_name}' not found at {skill_path}]"

        raw_content = skill_path.read_text(encoding="utf-8")
        body = raw_content
        if raw_content.startswith("---"):
            parts = raw_content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2]
        body = body.strip()

        def replace_ref(match: re.Match[str]) -> str:
            rel_path = match.group(1).strip()
            ref_path = skill_dir / rel_path
            if not ref_path.exists():
                ref_path = self.root.parent.parent.parent.parent / rel_path
            if ref_path.exists():
                return (
                    f"\n>>> REFERENCE: {ref_path.name} <<<\n"
                    f"{ref_path.read_text(encoding='utf-8')}\n"
                    ">>> END REFERENCE <<<\n"
                )
            return f"[System Warning: Referenced file not found: {rel_path}]"

        return re.sub(r"\{\{LOAD:(.*?)\}\}", replace_ref, body)


skill_registry = SkillRegistry()
