from pathlib import Path


def test_expected_project_folders_exist():
    root = Path(__file__).resolve().parents[1]
    for folder in ["data", "docs", "screenshots", "src", "tests"]:
        assert (root / folder).exists()


def test_flowchart_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "screenshots" / "flowchart.png").exists()
