# Command reference

Install with pip install seedl. Both seedl and
`python -m seedl` run the same standard-library CLI. No extras are needed.

| Command | Result |
| --- | --- |
| `seedl` | Compact guide and a first Python example |
| `seedl --help` | Commands and global options |
| `seedl --version` | Installed package version |
| `seedl help` | Python visualization APIs by task |
| `seedl help see` | Current signature, docstring and a complete Python example |
| `seedl list` | Categorized capabilities |
| `seedl doctor` | Python support range, package version, platform and optional package presence |
| `seedl examples` | Short complete examples to paste into Python |
| `seedl examples evaluate` | Complete scikit-learn training/evaluation example (printed, not executed) |
| seedl examples learn | Complete interactive learning simulation example |
| `seedl help evaluate` | New ML API, split provenance, metrics and captured plots |
| seedl help learn | Documentation for interactive educational simulations |
| `seedl examples attention` | One focused example; `--help` lists topics |
| seedl learn | Browse available interactive ML/DL learning lessons (same as seedl learn list) |
| seedl learn list | List all 15 classical ML and deep learning interactive lessons |
| seedl learn <concept> | Export interactive lesson offline (default: <concept>.html) |
| seedl learn <concept> --output <file> | Export lesson to a specified .html or .svg file |
| `seedl demo` | Create `seedl-demo.html` using authentic labeled example data |
| `seedl demo --output iris.svg` | Create a new SVG instead |

Use `--plain` before or after the command to disable colors. `NO_COLOR` and
non-TTY streams also disable colors. Output falls back to ASCII escapes when
the terminal encoding cannot represent a symbol. Commands run in PowerShell,
Linux shells, CI and notebooks' shell cells; Python snippets run in Python.

The CLI never reads an in-memory Python variable. To inspect your own tensor:

```python
import seedl as sd

# tensor is your existing Python object
# view = sd.see(tensor)
# view.save("tensor.html")
print(sd.guide("see"))
```

Help, list, examples and doctor are read-only. Doctor checks package presence
and distribution metadata without importing optional frameworks or allocating
GPU memory. A found package can still have incompatible binaries or drivers;
doctor does not claim to test that. Missing extras are optional and do not make
doctor fail. Python versions outside 3.10–3.13 are labeled outside the tested range.

Demo creates only its explicitly requested file (and parent directories),
refuses to overwrite an existing file, makes no network requests and does not
launch a browser. Open the resulting file yourself. Invalid commands or export
paths exit with status 2 and actionable guidance.

## Demo provenance

The bundled excerpt contains the first five Iris-setosa observations from
[Fisher's Iris dataset, UCI](https://archive.ics.uci.edu/dataset/53/iris),
licensed CC BY 4.0. Measurements in centimeters, in this column order:
sepal length, sepal width, petal length, petal width.

```text
5.1  3.5  1.4  0.2
4.9  3.0  1.4  0.2
4.7  3.2  1.3  0.2
4.6  3.1  1.5  0.2
5.0  3.6  1.4  0.2
```

Values were checked against UCI's `iris.data` during the audit. This is a small
example excerpt, not a representative statistical sample. Source, subset,
column meanings and units are included in the generated output.

## Learn command and simulations

seedl learn generates self-contained, offline interactive visualizations of fundamental machine learning and deep learning concepts.

- seedl learn list: Prints all supported classical ML and deep learning concepts grouped by domain.
- seedl learn <concept>: Generates an offline HTML lesson (e.g. seedl learn linear_regression -> linear_regression.html).
- seedl learn <concept> --output path/to/file.svg: Exports as standalone SVG with embedded typography.

Like seedl demo, seedl learn never overwrites existing files, makes zero network requests, and requires zero third-party dependencies (pure standard library and mathematical generation).
