from pathlib import Path


def screenshot_paths(project_root: Path) -> list[Path]:
    """Return available screenshot artifacts."""
    return sorted((project_root / "screenshots").glob("*"))
