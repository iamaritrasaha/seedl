# Bundled SeeDL typography

Google Sans is the UI/documentation family; JetBrains Mono is the monospace
family. Both are distributed under SIL OFL 1.1; see the adjacent license files.
Fonts are bundled to keep exported HTML/SVG and the website offline.

Sources fetched 2026-11-07:

- [Google Sans](https://github.com/google/fonts/tree/main/ofl/googlesans),
  `GoogleSans[GRAD,opsz,wght].ttf`; upstream project source revision
  `5c8f512c5fb42a7be099b951a7c097f96209c78e` (Google Sans v14.000).
- [JetBrains Mono](https://github.com/google/fonts/tree/main/ofl/jetbrainsmono),
  `JetBrainsMono[wght].ttf`; upstream project source revision
  `19371302b95d218af43299bce79ddbddd0bc364d`.

These are subsets of the normal variable fonts, retaining their names, license
metadata and weight axes. WOFF2 is used in browser exports; TTF is used by the
optional Matplotlib backend. Subsetting/compression used FontTools and Brotli
as development tools; neither is a SeeDL runtime dependency.

Included Unicode ranges:
`0000-024F, 0370-03FF, 0400-052F, 1E00-1EFF, 2000-206F, 2070-209F,
20A0-20CF, 2100-214F, 2190-22FF, 2300-23FF, 2500-259F, 27C0-27EF,
2980-29FF, 2A00-2AFF, FB00-FB06, FE00-FE0F, FFFD`.
Only glyphs available in the source font can be retained. Other writing systems
and unsupported mathematical glyphs use the browser/platform fallback; no
labels or numerical data are discarded.

The website's `assets/fonts/` copies must match these WOFF2 files and licenses.
Terminal emulators and third-party Markdown hosts control their own fonts;
SeeDL does not change a user's terminal settings or a GitHub page's CSS.
