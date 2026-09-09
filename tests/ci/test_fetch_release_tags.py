from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


_REPO = Path(__file__).resolve().parents[2]
_FETCH_TAGS = _REPO / "scripts" / "sandbox" / "fetch-release-tags.sh"
_PICK_TAGS = _REPO / "scripts" / "sandbox" / "pick-release-tags.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "install-e2e.yml"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def _init_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    _git(path, "config", "user.name", "Install E2E Test")
    _git(path, "config", "user.email", "install-e2e-test@example.invalid")
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "test: seed fixture")


def test_fetch_release_tags_populates_tagless_fork_from_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    fork = tmp_path / "fork"
    _init_repo(parent)
    _git(parent, "tag", "v2026.8.1")
    _git(parent, "tag", "v2026.8.31")
    _init_repo(fork)

    assert _git(fork, "tag", "--list").stdout == ""

    fetched = subprocess.run(
        [
            str(_FETCH_TAGS),
            "--repo",
            str(fork),
            "--remote",
            str(parent),
        ],
        text=True,
        capture_output=True,
    )

    assert fetched.returncode == 0, fetched.stderr
    assert _git(fork, "tag", "--list", "v*").stdout.splitlines() == [
        "v2026.8.1",
        "v2026.8.31",
    ]

    picked = subprocess.run(
        [str(_PICK_TAGS), "--count", "2", "--repo", str(fork)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert picked.stdout == '["v2026.8.1","v2026.8.31"]\n'


def test_install_e2e_fetches_parent_tags_before_building_matrix() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["pick-releases"]["steps"]

    sparse_paths = steps[0]["with"]["sparse-checkout"].splitlines()
    assert "scripts/sandbox/fetch-release-tags.sh" in sparse_paths
    assert "scripts/sandbox/pick-release-tags.sh" in sparse_paths

    pick_script = steps[1]["run"]
    fetch = "scripts/sandbox/fetch-release-tags.sh"
    pick = "scripts/sandbox/pick-release-tags.sh"
    assert fetch in pick_script
    assert pick_script.index(fetch) < pick_script.index(pick)
