# Typography

Use **Google Sans** for UI and documentation and **JetBrains Mono** for code and
monospace data. The same convention is recorded in the repository's `AGENTS.md`.

The website self-hosts WOFF2 files in `website/assets/fonts/`. Generated HTML/SVG
embeds them as data URLs, including font licenses. The optional Matplotlib
renderer registers the bundled TTF files. No CDN, required font service, new
runtime dependency, backend server or user-installed font is needed.

Font files are embedded at export/notebook display boundaries; cached raw SVG
figures stay compact and reference the same family names. Font loading is lazy and cached; discovery and plain-text summaries do not read
font assets. The bundled Latin, Greek, Cyrillic and scientific-symbol subsets
keep exports compact. Unsupported glyphs retain platform fallback rather than
changing labels. See [font provenance and licenses](../src/seedl/assets/fonts/README.md).

Terminal emulators and hosted Markdown pages such as GitHub control their own
fonts; the library cannot override them. All SeeDL-controlled web documentation
and visual UI use the shared families. Publication of the website is unchanged
and remains a separate, manually triggered action.
