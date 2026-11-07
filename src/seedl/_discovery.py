"""One lightweight catalog for Python and terminal discovery."""

from __future__ import annotations

import inspect
import textwrap

DOCS = "https://github.com/iamaritrasaha/seedl/blob/main/docs"
GROUPS = {
    "Start here": ("see", "compare", "learn"),
    "Networks and training": (
        "activations",
        "gradients",
        "gradient_health",
        "trace",
        "flow",
        "watch",
        "track",
        "compare_runs",
        "loss_landscape",
    ),
    "Vision and attention": (
        "attention",
        "attention_layers",
        "attention_rollout",
        "vit_attention",
        "features",
        "feature_maps",
        "convolution",
        "embeddings",
    ),
    "Data and predictions": (
        "dataset",
        "dataset_health",
        "dataset_drift",
        "predictions",
        "compare_predictions",
        "explain",
        "compare_attributions",
    ),
    "Classical machine learning": (
        "evaluate",
        "inspect_estimator",
        "feature_importance",
        "partial_dependence",
        "cv_results",
        "learning_curve",
        "validation_curve",
    ),
    "Quantum": (
        "bloch",
        "circuit",
        "statevector",
        "density_matrix",
        "entanglement",
        "quantum_noise",
    ),
    "Reports and saved analyses": ("report", "save_analysis", "load_analysis"),
}
EXAMPLES = {
    "evaluate": 'from sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nimport seedl as sd\n\nX, y = load_iris(return_X_y=True)\nX_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)\nmodel = LogisticRegression(max_iter=500).fit(X_train, y_train)  # your training code\nresult = sd.evaluate(model, X_test, y_test, split="test", backend="svg")\nresult.save("evaluation.html")',
    "see": 'import seedl as sd\n\nview = sd.see([[1, 2], [3, 4]], backend="svg")\nview.show()\nview.save("result.html")',
    "tensor": 'import numpy as np\nimport seedl as sd\n\ntensor = np.arange(24).reshape(2, 3, 4)  # synthetic example\nview = sd.see(tensor, interactive=True, dim_names=["batch", "row", "column"])\nview.save("tensor.html")',
    "network": 'import torch\nimport seedl as sd\n\ntorch.manual_seed(0)\nmodel = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())\nx = torch.ones(2, 4)  # synthetic input\nview = sd.see(model, input=x, backend="svg")\nview.save("network.html")',
    "activations": 'import torch\nimport seedl as sd\n\nmodel = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())\nview = sd.activations(model, input=torch.ones(2, 4), backend="svg")\nview.save("activations.html")',
    "gradients": 'import torch\nimport seedl as sd\n\nmodel = torch.nn.Linear(4, 1)\nmodel(torch.ones(2, 4)).sum().backward()  # your training code\nview = sd.gradients(model, backend="svg")\nview.save("gradients.html")',
    "gradient_health": 'import torch\nimport seedl as sd\n\nmodel = torch.nn.Linear(4, 1)\nmodel(torch.ones(2, 4)).sum().backward()\nview = sd.gradient_health(model, backend="svg")\nview.save("gradient-health.html")',
    "attention": 'import seedl as sd\n\nweights = [[0.8, 0.2], [0.3, 0.7]]  # synthetic routing probabilities\nview = sd.attention(weights, tokens=["query", "key"], backend="svg")\nview.save("attention.html")',
    "bloch": 'import seedl as sd\n\nview = sd.bloch("+", backend="svg")  # analytic single-qubit |+> state\nview.save("bloch.html")',
    "statevector": 'import seedl as sd\n\nview = sd.statevector([2**-0.5, 0, 0, 2**-0.5], backend="svg")\nview.save("bell-state.html")',
    "circuit": 'from qiskit import QuantumCircuit\nimport seedl as sd\n\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nview = sd.circuit(qc, backend="svg")\nview.save("circuit.html")',
}


def capabilities(*, width: int = 88) -> str:
    from . import api

    lines = ["SeeDL Python APIs", "", "Use these inside Python: import seedl as sd", ""]
    for group, names in GROUPS.items():
        lines.append(group)
        for name in names:
            doc = inspect.getdoc(getattr(api, name)) or ""
            description = doc.splitlines()[0]
            lines.extend(
                textwrap.wrap(
                    description,
                    width=width,
                    initial_indent=f"  sd.{name}  ",
                    subsequent_indent="    ",
                )
            )
        lines.append("")
    lines.extend(("Learn more: seedl help see", f"Examples: {DOCS}/GETTING_STARTED.md"))
    return "\n".join(lines)


def function_help(name: str) -> str:
    from . import api

    name = name.removeprefix("sd.").removeprefix("seedl.")
    if name not in {n for names in GROUPS.values() for n in names}:
        import difflib

        matches = difflib.get_close_matches(
            name, [n for names in GROUPS.values() for n in names], n=1
        )
        suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(f"Unknown Python API {name!r}.{suggestion} Run 'seedl list'.")
    function = getattr(api, name)
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    call_signature = signature.replace(
        parameters=[p.replace(annotation=inspect.Parameter.empty) for p in parameters],
        return_annotation=inspect.Signature.empty,
    )
    syntax = f"sd.{name}{call_signature}"
    if len(syntax) > 88:
        parts = []
        keyword_separator = False
        for param in call_signature.parameters.values():
            if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_separator:
                parts.append("    *,")
                keyword_separator = True
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                keyword_separator = True
            parts.append(f"    {param},")
        syntax = f"sd.{name}(\n" + "\n".join(parts) + "\n)"
    lines = [syntax, "", inspect.getdoc(function) or ""]
    example = EXAMPLES.get(name)
    if example:
        lines.extend(("", "Runnable Python example", "", example))
    reference = (
        "CLASSICAL_ML.md" if name in GROUPS["Classical machine learning"] else "REFERENCE.md"
    )
    lines.extend(("", f"Reference: {DOCS}/{reference}", f"Examples: {DOCS}/GETTING_STARTED.md"))
    return "\n".join(lines)


def guide(topic: str | None = None) -> str:
    """Return a printable Python API guide or help for one function.

    Examples: ``print(sd.guide())`` and ``print(sd.guide('see'))``.
    This only describes APIs; it does not execute examples or inspect variables.
    """
    return function_help(topic) if topic else capabilities()
