from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def data_artifact_exists(filename: str) -> bool:
    """Check whether a project data/model artifact exists."""
    return (DATA_DIR / filename).exists()
