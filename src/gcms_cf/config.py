from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    data_raw: Path
    data_interim: Path
    data_output: Path
    cache_dir: Path

def default_paths(project_root: str | Path):
    root = Path(project_root).resolve()
    return PipelinePaths(
        project_root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_output=root / "data" / "output",
        cache_dir=root / ".cache_pipeline",
    )
