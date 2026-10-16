# SeeDL 0.5.0 — Maintainer Release Checklist

This document provides the authoritative, step-by-step checklist for repository maintainers to finalize and publish the **SeeDL 0.5.0 (Neural Observatory)** release after the release preparation Pull Request is reviewed and merged into `main`.

> [!IMPORTANT]
> **Safety Guardrail:** Do not execute release tagging or publication until the release PR has been approved and merged into `main`, and all CI checks on `main` have passed cleanly.
> PyPI files are **immutable**; once uploaded, version `0.5.0` cannot be re-uploaded or modified.

---

## Pre-Release Verification (Before Merging PR)

- [ ] **1. Review Release Pull Request**
  - Verify that the PR title is `chore(release): prepare SeeDL 0.5.0 Neural Observatory`.
  - Confirm version metadata is bumped from `0.4.0` to `0.5.0` across:
    - `pyproject.toml` (`version = "0.5.0"`)
    - `src/seedl/__init__.py` (`__version__ = "0.5.0"`)
    - `src/seedl/_version.py` (`__version__ = "0.5.0"`)
    - `uv.lock`
  - Confirm release notes exist at `docs/RELEASE_NOTES_0.5.0.md`.
  - Verify all 6 GitHub Actions CI matrix checks pass on the PR:
    - Lint and Type Check (Ruff, Mypy, Lockfile)
    - Python 3.10 Test Matrix (`--extra all` & core zero-dependency)
    - Python 3.11 Test Matrix (`--extra all` & core zero-dependency)
    - Python 3.12 Test Matrix (`--extra all` & core zero-dependency)
    - Python 3.13 Test Matrix (`--extra all` & core zero-dependency)
    - Package Validation (Twine strict, `verify_release.py`, clean wheel/sdist install)

---

## Release Execution Steps (Post-Merge)

### Step 1: Merge PR & Sync Local Working Copy

- [ ] **Merge the Release PR into `main`**
  - Use GitHub web interface or `gh pr merge --merge` (or squash according to maintainer preference).
- [ ] **Pull latest `main` locally and verify clean state**
  ```bash
  git checkout main
  git pull origin main
  git status  # Must be clean
  ```
- [ ] **Verify commit history and HEAD**
  ```bash
  git log -n 5 --oneline
  ```

### Step 2: Local Pre-Tag Sanity Check

- [ ] **Run the release verification script against target tag `v0.5.0`**
  ```bash
  uv run python scripts/verify_release.py --tag v0.5.0
  ```
- [ ] **Build distributions and verify packaging**
  ```bash
  uv build --no-sources
  uv run --with twine twine check --strict dist/*.whl dist/*.tar.gz
  uv run python scripts/verify_release.py --check-dist --tag v0.5.0
  ```

### Step 3: Create and Push Signed Git Tag

- [ ] **Create an annotated (preferably GPG-signed) git tag**
  ```bash
  git tag -s v0.5.0 -m "Release SeeDL 0.5.0 — Neural Observatory"
  # (If GPG signing is not configured, use annotated tag: git tag -a v0.5.0 -m "Release SeeDL 0.5.0 — Neural Observatory")
  ```
- [ ] **Verify the tag points to the merge commit on `main`**
  ```bash
  git show v0.5.0
  ```
- [ ] **Push the tag to GitHub**
  ```bash
  git push origin v0.5.0
  ```

### Step 4: Publish GitHub Release (Triggers Automated PyPI Publishing)

- [ ] **Draft the GitHub Release**
  - Navigate to: `https://github.com/iamaritrasaha/seedl/releases/new`
  - **Tag version:** Select existing tag `v0.5.0`.
  - **Target:** `main`.
  - **Release title:** `SeeDL 0.5.0 — Neural Observatory`.
  - **Release description:** Copy contents from `docs/RELEASE_NOTES_0.5.0.md`.
    > [!IMPORTANT]
    > **Status Header Update:** When pasting the release notes into the GitHub Release form, replace the `**Status:** Prepared for release; publication pending.` line with `**Status:** Published`. Ensure the public release does not display an unpublished or pending notice.
  - **Attach binaries (optional):** Attach the built `dist/seedl-0.5.0-py3-none-any.whl` and `dist/seedl-0.5.0.tar.gz`.
- [ ] **Publish Release**
  - Click **Publish release**.
  > [!NOTE]
  > Publishing a GitHub release automatically triggers the workflow `.github/workflows/publish.yml`, which builds the distributions and publishes them to PyPI using Trusted Publishing (OIDC).

### Step 5: Monitor PyPI Trusted Publishing Workflow

- [ ] **Watch GitHub Actions workflow**
  - Visit: `https://github.com/iamaritrasaha/seedl/actions/workflows/publish.yml`
  - Verify that the `build` job passes (tests, lints, package validation).
  - Verify that the `publish` job exchanges the OIDC token with PyPI and completes upload successfully.

### Step 6: Verify Package on PyPI

- [ ] **Inspect PyPI project page**
  - Visit: `https://pypi.org/project/seedl/0.5.0/`
  - Confirm version `0.5.0` is live and metadata/README renders accurately.
- [ ] **Perform clean-room smoke test from PyPI**
  ```bash
  python -m venv /tmp/seedl_pypi_verify
  source /tmp/seedl_pypi_verify/bin/activate
  pip install --no-cache-dir seedl==0.5.0
  python -c "import seedl; print(seedl.__version__); assert seedl.__version__ == '0.5.0'"
  python -c "import seedl as sd; print(sd.see([1, 2, 3]))"
  deactivate
  rm -rf /tmp/seedl_pypi_verify
  ```

---

## Post-Release Tasks

### Step 7: Separate Website & Documentation PR (Post-Release)

> [!IMPORTANT]
> **Manual Deployment & Isolation Guardrail:**
> GitHub Pages deployment is **strictly manual** via GitHub Actions `workflow_dispatch` in `.github/workflows/pages.yml`. Pushing changes to `main` does **not** automatically deploy the website.
> Because `.github/workflows/pages.yml` installs published SeeDL packages from PyPI using `uv pip install`, website updates must be developed and merged in a **separate dedicated Pull Request** after `seedl==0.5.0` is confirmed live on PyPI.

Create a separate website update branch (e.g. `feature/website-0.5.0`) and pull request containing the following steps:

- [ ] **1. Update pinned package version in workflow**
  - In `.github/workflows/pages.yml`, update line 29:
    `uv pip install 'seedl==0.4.0' 'plotly==7.1.0' 'qiskit==2.5.2'` $\to$ `uv pip install 'seedl==0.5.0' 'plotly==7.1.0' 'qiskit==2.5.2'`
  - Update assertion in line 30: `assert seedl.__version__ == '0.5.0'`
- [ ] **2. Update `website/generate.py`**
  - Update generated demonstration assets using real SeeDL 0.5.0 APIs.
  - Run `python website/generate.py` to regenerate static and interactive assets.
- [ ] **3. Update `website/check.py`**
  - Update static link, provenance, and example verification checks.
  - Run `python website/check.py` to ensure all checks pass.
- [ ] **4. Update website documentation**
  - Update `website/docs/` files to document new 0.5.0 APIs and capabilities.
  - Update version badges across `website/index.html` and `website/playground.html` to `0.5.0`.
- [ ] **5. Update playground demos using real SeeDL 0.5.0 APIs**
  - Incorporate Neural Observatory interactive demos (e.g., execution flow, attention rollout, gradient health) into `website/playground.html`.
- [ ] **6. Preserve existing website redesign improvements**
  - Maintain PR #15 design enhancements, accessibility landmarks, theme tokens, and typography.
- [ ] **7. Verify actual browser interactions**
  - Run Playwright browser interaction verification:
    ```bash
    python scripts/verify_website.py
    ```
  - Confirm 100% test pass rate across all suites.
- [ ] **8. Pass CI on the Website PR**
  - Ensure all GitHub Actions CI checks on the website PR pass.
- [ ] **9. Manual merge by maintainer**
  - Review and merge the website PR into `main`.
- [ ] **10. Manual deployment through Pages workflow**
  - Navigate to **Actions** $\to$ **Official website (manual deployment)** (`.github/workflows/pages.yml`).
  - Click **Run workflow** on branch `main` to trigger deployment.
  - Confirm deployment succeeds at `https://iamaritrasaha.github.io/seedl/`.

### Step 8: Community Announcement & Communication

- [ ] **Announce release**
  - Share release notes on relevant community channels, GitHub Discussions, social channels, and research forums.
  - Celebrate the milestone with contributors!
