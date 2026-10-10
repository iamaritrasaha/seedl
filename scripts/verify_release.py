#!/usr/bin/env python3
"""Release verification script for SeeDL.

Validates that:
1. pyproject.toml contains all required release metadata.
2. Package version in pyproject.toml matches src/seedl/__init__.py.
3. Git release tag (if provided) matches the version (v<version>).
4. Built wheel and sdist (if checked) exist and match the version.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

# Use standard library tomllib (Python 3.11+) or tomli fallback
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print(
            "Error: tomllib (Python 3.11+) or tomli is required to run verify_release.py",
            file=sys.stderr,
        )
        sys.exit(1)

SEMVER_PATTERN = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:([a-zA-Z0-9._-]+))?$")


def verify_release(
    repo_root: Path,
    expected_tag: str | None = None,
    check_dist: bool = False,
) -> None:
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise RuntimeError(f"Missing pyproject.toml at {pyproject_path}")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    name = project.get("name")
    version = project.get("version")
    description = project.get("description")
    readme = project.get("readme")
    license_val = project.get("license")
    requires_python = project.get("requires-python")
    urls = project.get("urls", {})

    print(f"Checking project metadata for '{name}'...")

    # 1. Validate project name
    if name != "seedl":
        raise ValueError(f"Expected project name 'seedl', found '{name}'")

    # 2. Validate version format
    if not version or not isinstance(version, str):
        raise ValueError("Missing or invalid 'project.version' in pyproject.toml")
    if not SEMVER_PATTERN.match(version):
        raise ValueError(
            f"Version '{version}' in pyproject.toml does not follow semver/PEP 440 (e.g. 0.1.0)"
        )

    # 3. Validate required metadata
    if not description or not isinstance(description, str):
        raise ValueError("Missing 'project.description' in pyproject.toml")
    if readme != "README.md" or not (repo_root / "README.md").is_file():
        raise ValueError("Missing or invalid 'project.readme' pointing to README.md")
    if license_val != "MIT" or not (repo_root / "LICENSE").is_file():
        raise ValueError("Missing or invalid 'project.license' (expected 'MIT' with LICENSE file)")
    if not requires_python or not requires_python.startswith(">="):
        raise ValueError("Missing or invalid 'project.requires-python' in pyproject.toml")

    if not urls.get("Repository"):
        raise ValueError("Missing 'Repository' URL in pyproject.toml [project.urls]")
    if not urls.get("Issues"):
        raise ValueError("Missing 'Issues' URL in pyproject.toml [project.urls]")

    print(f"  pyproject.toml version: {version}")

    # 4. Check src/seedl/__init__.py version
    init_path = repo_root / "src" / "seedl" / "__init__.py"
    if not init_path.is_file():
        raise RuntimeError(f"Missing __init__.py at {init_path}")

    init_text = init_path.read_text(encoding="utf-8")
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_text)
    if not version_match:
        raise ValueError(f"Could not find __version__ in {init_path}")
    init_version = version_match.group(1)
    if init_version != version:
        raise ValueError(
            f"Version mismatch: pyproject.toml has '{version}', but src/seedl/__init__.py has '{init_version}'"
        )
    print(f"  src/seedl/__init__.py __version__: {init_version} (MATCH)")

    # 5. Check git tag if supplied
    if expected_tag:
        # Strip refs/tags/ if passed from GITHUB_REF
        clean_tag = expected_tag.removeprefix("refs/tags/")
        expected_tag_name = f"v{version}"
        if clean_tag != expected_tag_name:
            raise ValueError(
                f"Release tag mismatch: git tag is '{clean_tag}', but expected '{expected_tag_name}' for version {version}"
            )
        print(f"  Release tag: {clean_tag} matches expected '{expected_tag_name}' (MATCH)")

    # 6. Check distributions if requested
    if check_dist:
        dist_dir = repo_root / "dist"
        expected_whl = dist_dir / f"seedl-{version}-py3-none-any.whl"
        expected_sdist = dist_dir / f"seedl-{version}.tar.gz"

        if not expected_whl.is_file():
            raise FileNotFoundError(f"Missing built wheel at {expected_whl}")
        if not expected_sdist.is_file():
            raise FileNotFoundError(f"Missing built sdist at {expected_sdist}")

        print(f"  Wheel found: {expected_whl.name}")
        print(f"  Sdist found: {expected_sdist.name}")

        # Verify wheel internal metadata
        with zipfile.ZipFile(expected_whl) as whl:
            metadata_name = f"seedl-{version}.dist-info/METADATA"
            if metadata_name not in whl.namelist():
                raise ValueError(f"Wheel missing {metadata_name}")
            metadata_content = whl.read(metadata_name).decode("utf-8")
            if f"Name: {name}" not in metadata_content:
                raise ValueError(f"Wheel METADATA does not specify 'Name: {name}'")
            if f"Version: {version}" not in metadata_content:
                raise ValueError(f"Wheel METADATA does not specify 'Version: {version}'")

            # Check that py.typed exists in wheel
            if "seedl/py.typed" not in whl.namelist():
                raise ValueError("Wheel is missing 'seedl/py.typed'")

            # Check that tests and examples are NOT in wheel
            for fname in whl.namelist():
                if fname.startswith("tests/") or fname.startswith("examples/"):
                    raise ValueError(f"Wheel unexpectedly contains development file: {fname}")

        print("  Wheel contents and METADATA verified cleanly.")

    print("\nAll release verification checks passed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SeeDL release metadata and distributions.")
    parser.add_argument(
        "--tag",
        type=str,
        default=os.environ.get("GITHUB_REF_NAME"),
        help="Expected git release tag (e.g. v0.1.0 or GITHUB_REF_NAME)",
    )
    parser.add_argument(
        "--check-dist",
        action="store_true",
        help="Verify built wheel and sdist files in dist/",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to repository root",
    )

    args = parser.parse_args()

    try:
        verify_release(
            repo_root=args.repo_root,
            expected_tag=args.tag,
            check_dist=args.check_dist,
        )
    except Exception as exc:
        print(f"\n[RELEASE VERIFICATION FAILED]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
