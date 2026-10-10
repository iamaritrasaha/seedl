# SeeDL Publishing and Release Guide

This document describes how SeeDL is built, validated, and published to the [Python Package Index (PyPI)](https://pypi.org/project/seedl/) using secure OpenID Connect (OIDC) Trusted Publishing.

---

## 1. Overview & Architecture

### Packaging Model
- **Package Name:** `seedl` (on PyPI: [seedl](https://pypi.org/project/seedl/))
- **Build Backend:** `uv_build` (configured in [`pyproject.toml`](file:///media/hrik/Hrik/Projects/seedl/pyproject.toml))
- **Design Philosophy:** Lightweight, zero-dependency core. Basic inspection, scalar/sequence/matrix previews, educational simulations, and SVG rendering require only the Python standard library.
- **Optional Extras:**
  - `torch`: PyTorch model, tensor, activation, gradient, and explainability inspection.
  - `numpy`: NumPy array processing and accelerated SVD.
  - `plotly`: Interactive HTML visualizations.
  - `matplotlib`: Static PNG figures and publication charts.
  - `all`: All optional framework and plotting integrations.
  - `dev`: Testing and code quality tools (`pytest`, `ruff`, `mypy`).

---

## 2. GitHub Actions Workflows

The repository uses two separate workflows:

### A. Continuous Integration (`.github/workflows/ci.yml`)
- **Triggers:** Pull requests to `main`, pushes to `main`, and manual `workflow_dispatch`.
- **Permissions:** Read-only (`contents: read`). **Cannot publish packages.**
- **Jobs:**
  1. `lint-and-typecheck`: Verifies `uv.lock`, runs `ruff check`, `ruff format --check`, and `mypy src`.
  2. `test`: Multi-version Python test matrix (Python 3.10, 3.11, 3.12, 3.13 on `ubuntu-latest`). Tests both full extras (`--extra all`) and isolated core-only installations.
  3. `package-validation`: Builds distributions, runs `twine check --strict`, executes [`scripts/verify_release.py`](file:///media/hrik/Hrik/Projects/seedl/scripts/verify_release.py), and runs clean-room installation smoke tests for both the wheel and source distribution.

### B. PyPI Publishing (`.github/workflows/publish.yml`)
- **Trigger:** Exclusively triggered when a **GitHub Release is published** (`release: types: [published]`).
- **Security:**
  - Runs in two strictly isolated stages: `build` (read-only permissions) and `publish` (`id-token: write` restricted to the `pypi` GitHub environment).
  - Ordinary pushes and pull requests **cannot** trigger this workflow.
- **Jobs:**
  1. `build`: Checks out source, verifies lockfile, runs test suite and linters, validates that the release tag matches `pyproject.toml`, builds wheel and sdist, verifies distribution integrity with twine, conducts clean-room wheel and sdist installation tests, and uploads verified artifacts.
  2. `publish`: Downloads verified build artifacts, requests a short-lived OIDC token from GitHub, and uploads distributions via the official `pypa/gh-action-pypi-publish` action.

---

## 3. How PyPI Trusted Publishing (OIDC) Works

PyPI Trusted Publishing eliminates the need for permanent API tokens or hardcoded username/password credentials stored in GitHub Secrets.

```
+------------------------------------------------------------------------+
| 1. Maintainer creates GitHub Release with tag v0.1.0                   |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 2. GitHub Actions triggers .github/workflows/publish.yml               |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 3. Build Job runs tests, static checks, and builds wheel + sdist       |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 4. Publish Job enters "pypi" Environment, requests OIDC token via       |
|    id-token: write permission                                          |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 5. PyPI verifies the token claims:                                     |
|    - repository: iamaritrasaha/seedl                                   |
|    - workflow: .github/workflows/publish.yml                           |
|    - environment: pypi                                                 |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
| 6. PyPI grants a short-lived token and accepts package upload          |
+------------------------------------------------------------------------+
```

### PyPI Configuration Requirements
The pending publisher on PyPI must match these values exactly:
- **PyPI Project Name:** `SeeDL` (normalized package: `seedl`)
- **Owner:** `iamaritrasaha`
- **Repository:** `seedl`
- **Workflow name:** `publish.yml`
- **Environment:** `pypi`

---

## 4. Preparing a New Release

Follow these steps before tagging any release:

### Step 1: Version Consistency
Ensure the version is updated in both locations:
1. `pyproject.toml` (e.g. `version = "0.1.0"`)
2. `src/seedl/__init__.py` (e.g. `__version__ = "0.1.0"`)

### Step 2: Run Local Validations
Execute the full validation suite locally:

```bash
# 1. Check lockfile consistency
uv lock --check

# 2. Run automated test suite
uv run --extra dev --extra all pytest

# 3. Code formatting and linting
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .

# 4. Type checking
uv run --extra dev mypy

# 5. Build distribution packages
uv build --no-sources

# 6. Verify distributions and metadata
uv run --with twine twine check --strict dist/*.whl dist/*.tar.gz

# 7. Run release verification script
python scripts/verify_release.py --check-dist --tag v0.1.0
```

### Step 3: Commit and Push Changes
Commit version bumps, updated documentation, and release notes to `main`:

```bash
git add pyproject.toml src/seedl/__init__.py docs/
git commit -m "chore: prepare release v0.1.0"
git push origin main
```

---

## 5. Publishing the First Release (`v0.1.0`)

### Step 1: Verify GitHub Environment Settings
1. Go to your GitHub repository: `https://github.com/iamaritrasaha/seedl`
2. Navigate to **Settings** -> **Environments**.
3. Confirm that an environment named `pypi` exists.
4. *(Recommended)* Under **Deployment branches and tags**, select **Selected tags** and add rule `v*` to guarantee only tagged releases can deploy to PyPI.

### Step 2: Create the GitHub Release
1. Navigate to **Releases** -> **Draft a new release** (or `https://github.com/iamaritrasaha/seedl/releases/new`).
2. **Choose a tag:** Type `v0.1.0` and select **Create new tag: v0.1.0 on publish**.
3. **Target:** `main`.
4. **Release title:** `SeeDL 0.1.0`
5. **Description:** Copy the release notes from [`docs/RELEASE_NOTES_0.1.0.md`](file:///media/hrik/Hrik/Projects/seedl/docs/RELEASE_NOTES_0.1.0.md).
6. Click **Publish release**.

### Step 3: Monitor Workflow Execution
1. Navigate to the **Actions** tab on GitHub (`https://github.com/iamaritrasaha/seedl/actions`).
2. Open the **Publish SeeDL to PyPI** workflow run for tag `v0.1.0`.
3. Verify that:
   - The `build` job completes all tests, lint checks, and package builds.
   - The `publish` job exchanges the OIDC token with PyPI and completes package upload.

### Step 4: Verify on PyPI
1. Visit [https://pypi.org/project/seedl/0.1.0/](https://pypi.org/project/seedl/0.1.0/).
2. Confirm that version `0.1.0` is published and displays the README.
3. Test installation in a clean environment:

```bash
# Using pip
pip install seedl

# Or using uv
uv add seedl
```

4. Verify smoke test:

```python
import seedl as sd

view = sd.see([1, 4, 9, 16], backend="svg")
print(view.summary)
```

---

## 6. Troubleshooting & Recovery

### 1. PyPI OIDC Rejection ("Trusted Publisher exchange failed")
- **Cause:** Discrepancy between GitHub context and PyPI publisher configuration.
- **Remedy:** Check PyPI publisher settings:
  - Repository must be `iamaritrasaha/seedl` (case-sensitive).
  - Workflow filename must be `publish.yml` (not `.github/workflows/publish.yml` or `deploy.yml`).
  - Environment name must be `pypi`.
  - The release must be triggered from the repository owner account.

### 2. Build Job Fails on Tag Mismatch
- **Cause:** Release tag does not match `pyproject.toml` version (e.g. tag is `v0.1.1` but `pyproject.toml` has `0.1.0`).
- **Remedy:** Delete the incorrect tag, update `pyproject.toml` and `src/seedl/__init__.py`, commit, and create a matching tag.

### 3. Recovering from a Failed Publication
- PyPI enforces **immutable release files**: once a version number is uploaded to PyPI, files for that version can never be overwritten or re-uploaded.
- If the workflow failed **before** uploading to PyPI (e.g., during tests or twine check): fix the issue and re-run the failed GitHub Actions job.
- If files were partially uploaded or rejected by PyPI: bump the patch version (e.g., to `0.1.1`), push to `main`, and create a new release for `v0.1.1`.

---

## 7. Publishing Subsequent Versions

For future releases (`v0.2.0`, `v1.0.0`, etc.):
1. Bump version in `pyproject.toml` and `src/seedl/__init__.py`.
2. Update `uv.lock` with `uv lock`.
3. Add release notes in `docs/RELEASE_NOTES_<version>.md`.
4. Open a PR, merge to `main`, and publish a GitHub Release with tag `v<version>`.
