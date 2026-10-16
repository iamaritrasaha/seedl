"""Comprehensive security and offline-verification tests for SeeDL 0.5.0."""

from __future__ import annotations

import re

import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import seedl as sd
from seedl._dataset_health import dataset_drift_html
from seedl._gradient_health import gradient_health_html
from seedl._quantum_lab import quantum_noise_html
from seedl._reports import report
from seedl._trace_renderers import trace_html
from seedl._training_compare import run_comparison_html
from seedl._transformer_observatory import attention_rollout_html


def _assert_no_external_network_resources(html_content: str) -> None:
    """Strictly assert that HTML contains NO external script, link, or style references."""
    # Look for http:// or https:// or // URLs inside src or href attributes
    external_refs = re.findall(
        r'(?:src|href)=["\'](https?:)?//[^"\']+["\']', html_content, re.IGNORECASE
    )
    assert not external_refs, f"Found unauthorized external network references: {external_refs}"


def _assert_no_script_injection(
    html_content: str, payload: str = "<script>alert('pwned')</script>"
) -> None:
    """Assert that an adversarial script payload is NOT executed as an unescaped tag."""
    assert payload not in html_content, (
        "Raw malicious script payload found unescaped in HTML output!"
    )


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_security_flow_trace_malicious_module_name():
    """Module names containing HTML/script tags are safely escaped in flow HTML."""
    malicious_name = "<script>alert('xss_layer')</script>"

    class MaliciousModel(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Linear(2, 2)
            self.add_module(malicious_name, layer)

        def forward(self, x):
            return getattr(self, malicious_name)(x)

    model = MaliciousModel()
    trace = sd.trace(model, input=torch.randn(1, 2))
    flow_viz = sd.flow(trace)

    html_out = trace_html(flow_viz)
    _assert_no_external_network_resources(html_out)
    _assert_no_script_injection(html_out, malicious_name)
    assert (
        "&lt;script&gt;alert(&#x27;xss_layer&#x27;)&lt;/script&gt;" in html_out
        or "&lt;script&gt;alert('xss_layer')&lt;/script&gt;" in html_out
        or "\\u003cscript\\u003e" in html_out
    )


def test_security_attention_rollout_malicious_tokens():
    """Token strings with script breakout attempts are safely escaped in attention rollout HTML."""
    breakout_token = "</script><script>alert('breakout')</script>"
    tokens = ["[CLS]", breakout_token, "world"]
    matrices = [
        [
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ]
    ]

    viz = sd.attention_rollout(matrices, tokens=tokens)
    html_out = attention_rollout_html(viz)

    _assert_no_external_network_resources(html_out)
    # The literal raw string should not break the script tag
    assert "</script><script>alert('breakout')</script>" not in html_out
    assert "\\u003cscript\\u003e" in html_out or "&lt;script&gt;" in html_out


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_security_gradient_health_offline():
    """Gradient health HTML is fully offline and self-contained."""
    model = nn.Linear(3, 2)
    x = torch.randn(1, 3)
    loss = model(x).sum()
    loss.backward()

    viz = sd.gradient_health(model)
    html_out = gradient_health_html(viz)
    _assert_no_external_network_resources(html_out)
    assert "<!DOCTYPE html>" in html_out


def test_security_quantum_noise_html_offline_and_escaped():
    """Quantum noise explorer HTML is fully offline and resilient against malformed inputs."""
    viz = sd.quantum_noise(channel="depolarizing", param=0.3, steps=4)
    html_out = quantum_noise_html(viz)

    _assert_no_external_network_resources(html_out)
    assert "<!DOCTYPE html>" in html_out


def test_security_dataset_drift_html_offline():
    """Dataset drift HTML is completely self-contained with no external dependencies."""
    ref = [float(i) for i in range(100)]
    comp = [float(i + 2) for i in range(100)]
    viz = sd.dataset_drift(ref, comp)

    html_out = dataset_drift_html(viz)
    _assert_no_external_network_resources(html_out)


def test_security_run_comparison_html_offline():
    """Run comparison HTML is self-contained with no external dependencies."""
    run1 = {"step": [1, 2, 3], "loss": [0.9, 0.5, 0.2]}
    run2 = {"step": [1, 2, 3], "loss": [0.85, 0.45, 0.18]}
    viz = sd.compare_runs(run1, run2, name_a="Run A", name_b="Run B")

    html_out = run_comparison_html(viz)
    _assert_no_external_network_resources(html_out)


def test_security_report_builder_offline():
    """Report builder produces strictly offline documents."""
    rep = report("Offline Report", abstract="Testing zero CDN calls.")
    html_out = rep.to_html()
    _assert_no_external_network_resources(html_out)
