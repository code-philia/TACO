from __future__ import annotations

from rek.models import CandidateMatch


def format_candidates_as_agent_hint(
    issue: str,
    candidates: list[CandidateMatch],
    max_candidates: int = 5,
) -> str:
    if not candidates:
        return ""

    trimmed = candidates[:max(1, max_candidates)]
    lines: list[str] = []
    lines.append("### 🧠 Expert Navigation Hint (Highly Recommended)")
    lines.append(
        "An auxiliary expert system has analyzed the repository knowledge base and returned ranked suspicious locations for the current issue."
    )
    lines.append("")
    lines.append(f"**Issue Under Investigation**: {issue}")
    lines.append("")
    lines.append("**Most Suspicious Locations (Use as Hints, Not Final Answer)**:")
    lines.append("")

    for candidate in trimmed:
        symbol = candidate.symbol or "Entire File"
        rationale = candidate.rationale or "No rationale provided."
        lines.append(f"- **File**: `{candidate.file_path}`")
        lines.append(f"  - **Function/Class**: `{symbol}`")
        lines.append(f"  - **Rationale**: {rationale}")
        if candidate.score is not None:
            lines.append(f"  - **Score**: {candidate.score:.4f}")
        lines.append("")

    lines.append("---")
    lines.append("### 💡 Strategic Instructions for the Agent")
    lines.append(
        "1. Treat this hint as a high-recall navigation aid. You must still verify the real code before producing a final localization answer."
    )
    lines.append(
        "2. Prioritize the top-ranked files and functions first, but keep room for your own independently discovered candidates."
    )
    lines.append(
        "3. If your own search strongly contradicts this hint, investigate both paths and merge only the verified evidence."
    )
    lines.append(
        "4. Keep the final suspicious list short and confidence-ranked; prefer exact functions over broad file-only answers when possible."
    )
    return "\n".join(lines).strip()
