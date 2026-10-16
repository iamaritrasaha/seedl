"""The Scientific Report Builder: Combine multiple SeeDL views into offline, publication-grade interactive reports."""

from __future__ import annotations

import base64
import html
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from .exceptions import SeeDLError
from .view import Visualization

__version__ = "0.4.0"


@dataclass
class ReportSection:
    """A titled section of a scientific report containing one or more visualizations and narrative notes."""

    title: str
    views: list[Visualization] = field(default_factory=list)
    narrative: str = ""
    notes: list[str] = field(default_factory=list)


class ScientificReport:
    """A self-contained scientific report combining multiple SeeDL visualizations."""

    def __init__(
        self,
        title: str,
        views: list[Visualization] | list[ReportSection] | None = None,
        *,
        subtitle: str = "",
        abstract: str = "",
        author: str = "",
        affiliation: str = "",
        theme: str = "dark",  # "dark" or "light"
        max_size_mb: float = 25.0,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.abstract = abstract
        self.author = author
        self.affiliation = affiliation
        self.theme = theme
        self.max_size_mb = max_size_mb
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        self.sections: list[ReportSection] = []

        if views:
            if isinstance(views[0], ReportSection):
                self.sections = list(views)  # type: ignore[arg-type]
            else:
                # Wrap list of visualizations into default sections
                for idx, v in enumerate(views):  # type: ignore[union-attr]
                    sec_title = v.title or f"Figure {idx + 1}"
                    self.sections.append(ReportSection(title=sec_title, views=[v]))

    def add_section(
        self,
        title: str,
        views: list[Visualization] | Visualization,
        narrative: str = "",
        notes: list[str] | None = None,
    ) -> ScientificReport:
        """Add a new section to the report."""
        v_list = [views] if isinstance(views, Visualization) else list(views)
        self.sections.append(
            ReportSection(
                title=title,
                views=v_list,
                narrative=narrative,
                notes=notes or [],
            )
        )
        return self

    def to_html(self) -> str:
        """Generate a complete, self-contained offline interactive HTML report."""
        report_title = html.escape(self.title)
        report_subtitle = html.escape(self.subtitle)
        abstract_escaped = html.escape(self.abstract)
        author_escaped = html.escape(self.author)
        affil_escaped = html.escape(self.affiliation)
        timestamp_escaped = html.escape(self.timestamp)

        # Build TOC
        toc_items = []
        sections_html = []
        fig_counter = 1

        for s_idx, sec in enumerate(self.sections, 1):
            sec_id = f"section-{s_idx}"
            s_title = html.escape(sec.title)
            toc_items.append(f'<li><a href="#{sec_id}">{s_idx}. {s_title}</a></li>')

            narrative_html = ""
            if sec.narrative:
                narrative_html = f'<div class="narrative">{html.escape(sec.narrative)}</div>'

            figs_html = []
            for v in sec.views:
                fig_id = f"fig-{fig_counter}"
                v_title = html.escape(v.title)
                v_summary = html.escape(v.summary)

                # Render view figure: SVG or PNG
                chart_markup = ""
                try:
                    if v.backend == "matplotlib":
                        fig = v.figure
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        chart_markup = f'<img src="data:image/png;base64,{b64}" alt="{v_title}" style="max-width:100%; height:auto;" />'
                    elif v.kind == "execution_trace":
                        from ._trace_renderers import svg_trace

                        chart_markup = svg_trace(v)
                    elif v.kind == "attention_rollout":
                        from ._transformer_observatory import svg_attention_rollout

                        chart_markup = svg_attention_rollout(v)
                    elif v.kind == "attribution_comparison":
                        from ._explain_compare import svg_attribution_comparison

                        chart_markup = svg_attribution_comparison(v)
                    elif v.kind == "quantum_noise":
                        from ._quantum_lab import svg_quantum_noise

                        chart_markup = svg_quantum_noise(v)
                    elif v.kind == "gradient_health":
                        from ._gradient_health import svg_gradient_health

                        chart_markup = svg_gradient_health(v)
                    else:
                        from .renderers import render

                        svg_content = render(v, "svg")
                        chart_markup = svg_content if isinstance(svg_content, str) else str(v)
                except Exception as exc:
                    chart_markup = (
                        f'<div class="error-box">Render notice: {html.escape(str(exc))}</div>'
                    )

                notes_html = ""
                if v.notes:
                    notes_items = "".join(f"<li>{html.escape(n)}</li>" for n in v.notes)
                    notes_html = f'<ul class="fig-notes">{notes_items}</ul>'

                figs_html.append(
                    f"""
                    <div class="figure-card" id="{fig_id}">
                      <div class="figure-header">
                        <span class="fig-label">Figure {fig_counter}:</span> {v_title}
                      </div>
                      <div class="chart-wrapper">
                        {chart_markup}
                      </div>
                      <div class="figure-caption">
                        <strong>Summary:</strong> {v_summary}
                      </div>
                      {notes_html}
                    </div>
                    """
                )
                fig_counter += 1

            sections_html.append(
                f"""
                <section class="report-section" id="{sec_id}">
                  <h2><span class="sec-num">{s_idx}.</span> {s_title}</h2>
                  {narrative_html}
                  <div class="figures-group">
                    {"".join(figs_html)}
                  </div>
                </section>
                """
            )

        doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{report_title} — SeeDL Scientific Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
    }}
    .light-theme {{
      --bg: #f8fafc;
      --card: #ffffff;
      --card-alt: #f1f5f9;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #0284c7;
      --accent-hover: #0369a1;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      transition: background 0.2s, color 0.2s;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 260px 1fr;
      min-height: 100vh;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ display: none; }}
    }}
    .sidebar {{
      background: var(--card);
      border-right: 1px solid var(--border);
      padding: 24px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
    }}
    .sidebar h3 {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .toc {{
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 13px;
    }}
    .toc a {{
      color: var(--text);
      text-decoration: none;
      transition: color 0.15s;
    }}
    .toc a:hover {{ color: var(--accent); }}
    .main-content {{
      padding: 40px 48px;
      max-width: 1050px;
    }}
    header.report-header {{
      border-bottom: 2px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 32px;
    }}
    header h1 {{
      font-size: 30px;
      font-weight: 800;
      color: var(--accent);
      line-height: 1.25;
    }}
    .meta-line {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 8px;
    }}
    .theme-toggle {{
      float: right;
      background: var(--card-alt);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .report-section {{
      margin-bottom: 48px;
    }}
    .report-section h2 {{
      font-size: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-bottom: 16px;
      color: var(--text);
    }}
    .sec-num {{ color: var(--accent); }}
    .narrative {{
      font-size: 15px;
      margin-bottom: 20px;
      color: var(--text);
    }}
    .figures-group {{
      display: flex;
      flex-direction: column;
      gap: 28px;
    }}
    .figure-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }}
    .figure-header {{
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 12px;
      color: var(--text);
    }}
    .fig-label {{ color: var(--accent); }}
    .chart-wrapper {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      overflow-x: auto;
      margin-bottom: 12px;
    }}
    .figure-caption {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .fig-notes {{
      margin-top: 10px;
      padding-left: 20px;
      font-size: 12px;
      color: var(--muted);
    }}
    footer.report-footer {{
      margin-top: 60px;
      border-top: 1px solid var(--border);
      padding-top: 20px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    @media print {{
      body {{ background: #fff !important; color: #000 !important; }}
      .sidebar, .theme-toggle {{ display: none !important; }}
      .main-content {{ padding: 0 !important; max-width: 100% !important; }}
      .figure-card {{ break-inside: avoid; border: 1px solid #ccc !important; box-shadow: none !important; }}
      .chart-wrapper {{ background: #fff !important; }}
    }}
  </style>
</head>
<body class="{"" if self.theme == "dark" else "light-theme"}">
<div class="layout">
  <aside class="sidebar">
    <h3>Contents</h3>
    <ul class="toc">
      {"".join(toc_items)}
    </ul>
  </aside>

  <main class="main-content">
    <header class="report-header">
      <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
      <h1>{report_title}</h1>
      {f'<p style="font-size:16px; margin-top:6px; color:var(--text);">{report_subtitle}</p>' if report_subtitle else ""}
      <div class="meta-line">
        {f"<span>Author: {author_escaped}</span> · " if author_escaped else ""}
        {f"<span>{affil_escaped}</span> · " if affil_escaped else ""}
        <span>Generated: {timestamp_escaped}</span> ·
        <span>SeeDL v{__version__}</span>
      </div>
      {f'<div style="margin-top:16px; font-style:italic; font-size:14px; color:var(--muted);">{abstract_escaped}</div>' if abstract_escaped else ""}
    </header>

    {"".join(sections_html)}

    <footer class="report-footer">
      <span>Built with SeeDL {__version__} — Scientific AI Visualization</span>
      <span>Self-contained offline document · No external CDN calls</span>
    </footer>
  </main>
</div>

<script>
  function toggleTheme() {{
    document.body.classList.toggle('light-theme');
  }}
</script>
</body>
</html>
"""
        # Validate size bounds
        size_bytes = len(doc.encode("utf-8"))
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.max_size_mb:
            raise SeeDLError(
                f"Generated report payload ({size_mb:.1f} MB) exceeds maximum allowed size ({self.max_size_mb} MB)."
            )

        return doc

    def save(self, filepath: str | Path, *, format: str | None = None) -> Path:
        """Export report to self-contained HTML file."""
        path = Path(filepath)
        ext = (format or path.suffix).lower().lstrip(".")
        if ext in {"html", "htm", ""}:
            path = path.with_suffix(".html") if not path.suffix else path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_html(), encoding="utf-8")
            return path
        raise SeeDLError(f"Unsupported report export format {ext!r}; use .html.")

    def show(self) -> None:
        """Display report in notebook or print summary in terminal."""
        try:
            from IPython import get_ipython  # type: ignore[import-not-found]

            if get_ipython() is not None:
                from IPython.display import HTML, display  # type: ignore[import-not-found]

                display(HTML(self.to_html()))
                return
        except ImportError:
            pass
        print(f"ScientificReport: {self.title} ({len(self.sections)} section(s))")

    def _repr_html_(self) -> str:
        return self.to_html()


def report(
    title: str,
    views: list[Visualization] | list[ReportSection] | None = None,
    *,
    subtitle: str = "",
    abstract: str = "",
    author: str = "",
    affiliation: str = "",
    theme: str = "dark",
    max_size_mb: float = 25.0,
) -> ScientificReport:
    """Create a publication-grade scientific report combining multiple SeeDL visualizations.

    Parameters:
        title: Title of the scientific report.
        views: List of Visualization instances or ReportSection instances to include.
        subtitle: Optional subtitle or secondary description.
        abstract: Optional executive summary or abstract paragraph.
        author: Report author name or team.
        affiliation: Institutional or project affiliation.
        theme: UI theme ('dark' or 'light').
        max_size_mb: Memory budget in megabytes before warning/error.
    """
    return ScientificReport(
        title=title,
        views=views,
        subtitle=subtitle,
        abstract=abstract,
        author=author,
        affiliation=affiliation,
        theme=theme,
        max_size_mb=max_size_mb,
    )
