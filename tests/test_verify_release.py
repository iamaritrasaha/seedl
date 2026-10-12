"""Exercise the release CLI exactly as branch CI and publishing invoke it."""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release.py"


@pytest.fixture
def release_tree(tmp_path):
    for name in ("pyproject.toml", "README.md", "LICENSE", "src/seedl/__init__.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "seedl-0.2.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("seedl-0.2.0.dist-info/METADATA", "Name: seedl\nVersion: 0.2.0\n")
        wheel.writestr("seedl/py.typed", "")
    with tarfile.open(dist / "seedl-0.2.0.tar.gz", "w:gz") as archive:
        archive.add(tmp_path / "pyproject.toml", arcname="seedl-0.2.0/pyproject.toml")
    return tmp_path


def run_verifier(root, *args, ref=None):
    env = os.environ.copy()
    env.pop("GITHUB_REF_NAME", None)
    if ref is not None:
        env["GITHUB_REF_NAME"] = ref
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("ref", [None, "main", "feature/visuals"])
@pytest.mark.parametrize("args", [(), ("--check-dist",)])
def test_branch_verification_does_not_require_tag(release_tree, ref, args):
    result = run_verifier(release_tree, *args, ref=ref)
    assert result.returncode == 0, result.stderr
    assert "__version__: 0.2.0 (MATCH)" in result.stdout
    assert "Release tag:" not in result.stdout
    if args:
        assert "Wheel contents and METADATA verified" in result.stdout
        assert "Sdist found:" in result.stdout


@pytest.mark.parametrize("tag", ["v0.2.0", "refs/tags/v0.2.0"])
@pytest.mark.parametrize("args", [(), ("--check-dist",)])
def test_explicit_matching_release_tag(release_tree, tag, args):
    result = run_verifier(release_tree, *args, "--tag", tag, ref="main")
    assert result.returncode == 0, result.stderr
    assert "Release tag: v0.2.0 matches" in result.stdout


@pytest.mark.parametrize("tag", ["v0.1.0", "main", "", "refs/heads/v0.2.0"])
@pytest.mark.parametrize("args", [(), ("--check-dist",)])
def test_explicit_invalid_tags_rejected(release_tree, tag, args):
    result = run_verifier(release_tree, *args, "--tag", tag)
    assert result.returncode == 1
    assert "Release tag mismatch" in result.stderr


def test_metadata_checks_still_required_without_tag(release_tree):
    (release_tree / "src/seedl/__init__.py").write_text('__version__ = "0.9.9"')
    result = run_verifier(release_tree, ref="main")
    assert result.returncode == 1
    assert "Version mismatch" in result.stderr


@pytest.mark.parametrize("filename", ["seedl-0.2.0-py3-none-any.whl", "seedl-0.2.0.tar.gz"])
def test_missing_distributions_rejected_without_tag(release_tree, filename):
    (release_tree / "dist" / filename).unlink()
    result = run_verifier(release_tree, "--check-dist", ref="main")
    assert result.returncode == 1
    assert "Missing built" in result.stderr


def test_invalid_wheel_rejected_without_tag(release_tree):
    with zipfile.ZipFile(release_tree / "dist/seedl-0.2.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("seedl-0.2.0.dist-info/METADATA", "Name: seedl\nVersion: 0.9.9\n")
    result = run_verifier(release_tree, "--check-dist", ref="main")
    assert result.returncode == 1
    assert "Wheel METADATA" in result.stderr
