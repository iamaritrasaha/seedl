"""Bundled, licensed typography. Font files load only when rendering/exporting."""

from __future__ import annotations

import base64
from functools import lru_cache
from importlib.resources import files
from typing import Any

FONT = "Google Sans, sans-serif"
MONO = "JetBrains Mono, monospace"


@lru_cache(maxsize=1)
def font_css() -> str:
    rules = []
    assets = files("seedl").joinpath("assets").joinpath("fonts")
    for family, filename, weights in (
        ("Google Sans", "GoogleSans", "400 700"),
        ("JetBrains Mono", "JetBrainsMono", "100 800"),
    ):
        data = base64.b64encode(assets.joinpath(f"{filename}.woff2").read_bytes()).decode("ascii")
        license_text = (
            assets.joinpath(f"{filename}-OFL.txt").read_text(encoding="utf-8").replace("*/", "* /")
        )
        rules.append(
            f"/* {license_text} */\n@font-face{{font-family:'{family}';font-style:normal;font-weight:{weights};font-display:swap;src:url(data:font/woff2;base64,{data}) format('woff2')}}"
        )
    return "\n".join(rules)


def embed_fonts(markup: str) -> str:
    """Add offline fonts without changing data, scripts or viewport geometry."""
    if 'data-seedl-fonts="true"' in markup:
        return markup
    style = f'<style data-seedl-fonts="true">{font_css()}</style>'
    if "</head>" in markup:
        return markup.replace("</head>", style + "</head>", 1)
    if "</svg>" in markup and markup.lstrip().startswith("<svg"):
        # SVG is XML, including license comments inside its style element.
        style = style.replace("&", "&amp;")
        return markup.replace("</svg>", style + "</svg>", 1)
    return style + markup


def style_figure(figure: Any) -> Any:
    """Keep existing Plotly geometry/themes while setting the shared family."""
    if isinstance(figure, str):
        # Keep cached raw SVG compact; export/notebook boundaries embed fonts.
        return figure
    if hasattr(figure, "update_layout"):
        figure.update_layout(font_family=FONT)
    elif hasattr(figure, "findobj"):
        from matplotlib.text import Text

        _register_matplotlib_fonts()
        for text in figure.findobj(Text):
            family = (
                "JetBrains Mono"
                if any("mono" in name.lower() for name in text.get_fontfamily())
                else "Google Sans"
            )
            text.set_fontfamily(family)
    return figure


@lru_cache(maxsize=1)
def _register_matplotlib_fonts() -> None:
    from matplotlib import font_manager

    assets = files("seedl").joinpath("assets").joinpath("fonts")
    for filename in ("GoogleSans.ttf", "JetBrainsMono.ttf"):
        font_manager.fontManager.addfont(str(assets.joinpath(filename)))
