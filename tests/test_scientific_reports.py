"""Tests for the Scientific Report Builder in SeeDL 0.5.0."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from seedl._reports import ScientificReport, report
from seedl.exceptions import SeeDLError
from seedl.view import Visualization


def _create_sample_visualization(name: str = "Test Viz") -> Visualization:
    return Visualization(
        kind="dataset",
        title=name,
        summary="A test dataset summary",
        data={"shape": [10, 5], "means": [1.0, 2.0, 3.0, 4.0, 5.0]},
        metadata={"Samples": 10, "Features": 5},
        notes=["Note 1", "Note 2"],
    )


def test_scientific_report_basic_creation():
    """Build a report with multiple sections and verify table of contents."""
    v1 = _create_sample_visualization("Model Overview")
    v2 = _create_sample_visualization("Feature Analysis")

    rep = report(
        title="Neural Architecture Study",
        views=[v1, v2],
        subtitle="Experimental Run A",
        abstract="Investigating representation drift.",
        author="Research Team",
        affiliation="Institute of Science",
        theme="dark",
    )

    assert isinstance(rep, ScientificReport)
    assert len(rep.sections) == 2

    html_out = rep.to_html()
    assert "<!DOCTYPE html>" in html_out
    assert "Neural Architecture Study" in html_out
    assert "Research Team" in html_out
    assert "Institute of Science" in html_out
    assert "Contents" in html_out
    # Check TOC links to sections
    assert 'href="#section-1"' in html_out
    assert 'id="section-1"' in html_out
    assert 'href="#section-2"' in html_out
    assert 'id="section-2"' in html_out


def test_scientific_report_xss_sanitization():
    """Adversarial input with script tags is properly escaped in all fields."""
    v_malicious = _create_sample_visualization("<script>alert('xss_viz')</script>")

    rep = report(
        title="<script>alert('xss_title')</script>",
        views=[v_malicious],
        subtitle="<script>alert('xss_subtitle')</script>",
        abstract="<script>alert('xss_abstract')</script>",
        author="<script>alert('xss_author')</script>",
        affiliation="<script>alert('xss_affil')</script>",
    )
    rep.add_section(
        title="<script>alert('xss_sec')</script>",
        views=[],
        narrative="<script>alert('xss_narrative')</script>",
        notes=["<script>alert('xss_note')</script>"],
    )

    html_out = rep.to_html()

    # The raw unescaped script tag should NOT exist for any of these
    assert "<script>alert('xss_title')</script>" not in html_out
    assert "<script>alert('xss_viz')</script>" not in html_out
    assert "<script>alert('xss_subtitle')</script>" not in html_out
    assert "<script>alert('xss_abstract')</script>" not in html_out
    assert "<script>alert('xss_author')</script>" not in html_out
    assert "<script>alert('xss_affil')</script>" not in html_out
    assert "<script>alert('xss_sec')</script>" not in html_out
    assert "<script>alert('xss_narrative')</script>" not in html_out
    assert "<script>alert('xss_note')</script>" not in html_out

    # Instead, properly escaped entities should appear
    assert (
        "&lt;script&gt;alert(&#x27;xss_title&#x27;)&lt;/script&gt;" in html_out
        or "&lt;script&gt;alert('xss_title')&lt;/script&gt;" in html_out
    )


def test_scientific_report_offline_guarantee():
    """Verify that reports contain zero external network dependencies (no http:// or https:// links)."""
    v1 = _create_sample_visualization()
    rep = report("Offline Report", views=[v1])
    html_out = rep.to_html()

    # Check for script or link tags with external src/href
    lines = html_out.splitlines()
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("<script src=") or stripped.startswith(
            '<link rel="stylesheet" href='
        ):
            assert "http://" not in stripped
            assert "https://" not in stripped
            assert "//" not in stripped


def test_scientific_report_save_and_theme_toggle():
    """Verify saving to HTML file and theme toggle functionality."""
    v1 = _create_sample_visualization()
    rep = report("Exportable Report", views=[v1], theme="dark")

    html_out = rep.to_html()
    assert "toggleTheme()" in html_out

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "sub" / "report.html"
        saved = rep.save(out_path)
        assert saved.exists()
        assert saved.is_file()
        content = saved.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Exportable Report" in content


def test_scientific_report_size_budget_enforcement():
    """Verify that exceeding max_size_mb raises SeeDLError."""
    v1 = _create_sample_visualization()
    # Set a tiny max_size_mb (0.0001 MB = 100 bytes)
    rep = report("Tiny Report", views=[v1], max_size_mb=0.0001)

    with pytest.raises(SeeDLError, match="exceeds maximum allowed size"):
        rep.to_html()
