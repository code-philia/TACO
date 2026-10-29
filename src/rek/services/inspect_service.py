from pathlib import Path

from rek.config import resolve_elements_path
from rek.models import CommandMeta, InspectPathsRequest, InspectPathsResponse


class InspectService:
    def paths(self, request: InspectPathsRequest) -> InspectPathsResponse:
        resolved_elements_path = resolve_elements_path(
            request.kb_path,
            cli_value=request.elements_path,
        )
        return InspectPathsResponse(
            meta=CommandMeta(status="ok", message="Path inspection completed."),
            repo_path=request.repo_path,
            repo_exists=Path(request.repo_path).exists(),
            kb_path=request.kb_path,
            kb_exists=Path(request.kb_path).exists() if request.kb_path else None,
            elements_path=resolved_elements_path,
            elements_exists=(
                Path(resolved_elements_path).exists() if resolved_elements_path else None
            ),
        )
