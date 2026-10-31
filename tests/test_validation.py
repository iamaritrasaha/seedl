"""Regression tests for SeeDL validation logic and option dispatch."""

from __future__ import annotations

import math
from typing import Any

import pytest

from seedl._validation import (
    resolve_see_backend,
    validate_max_points,
    validate_object_arguments,
    validate_see_options,
    validate_tensor_view_options,
)
from seedl.exceptions import SeeDLError


class _DummyModule:
    pass


class _DummyTensor1D:
    shape = (10,)


class _DummyTensor2D:
    shape = (4, 4)


def test_validate_max_points_valid() -> None:
    validate_max_points(1)
    validate_max_points(2_048)
    validate_max_points(100_000)


@pytest.mark.parametrize("invalid", [0, -1, -100, True, False, 1.5, "100", None])
def test_validate_max_points_invalid(invalid: Any) -> None:
    with pytest.raises(ValueError, match="max_points must be a positive integer."):
        validate_max_points(invalid)


def test_validate_see_options_valid_defaults() -> None:
    validate_see_options(
        view="auto",
        theme="light",
        color_by="auto",
        detail="summary",
        max_neurons=100,
        max_connections=1500,
        sample=0,
        min_weight=0.0,
        edge_opacity=0.45,
    )


def test_validate_see_options_invalid_view() -> None:
    with pytest.raises(ValueError, match="view must be auto"):
        validate_see_options(
            view="invalid_view",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


def test_validate_see_options_invalid_theme() -> None:
    with pytest.raises(
        ValueError, match="theme must be light, dark, scientific, minimal, or presentation."
    ):
        validate_see_options(
            view="auto",
            theme="neon",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


def test_validate_see_options_invalid_color_by() -> None:
    with pytest.raises(ValueError, match="color_by must be auto, layer, activation, or gradient."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="rainbow",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


def test_validate_see_options_invalid_detail() -> None:
    with pytest.raises(ValueError, match="detail must be summary, full, or leaves."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="verbose",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


@pytest.mark.parametrize("invalid", [0, -5, 1001, True, False, 1.0])
def test_validate_see_options_invalid_max_neurons(invalid: Any) -> None:
    with pytest.raises(ValueError, match="max_neurons must be an integer between 1 and 1000."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=invalid,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


@pytest.mark.parametrize("invalid", [0, -10, 20001, True, False, "1500"])
def test_validate_see_options_invalid_max_connections(invalid: Any) -> None:
    with pytest.raises(ValueError, match="max_connections must be an integer between 1 and 20000."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=invalid,
            sample=0,
            min_weight=0.0,
            edge_opacity=0.45,
        )


@pytest.mark.parametrize("invalid", [-1, True, False, 1.5, "0"])
def test_validate_see_options_invalid_sample(invalid: Any) -> None:
    with pytest.raises(ValueError, match="sample must be a nonnegative integer."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=invalid,
            min_weight=0.0,
            edge_opacity=0.45,
        )


@pytest.mark.parametrize("invalid", [-0.01, math.nan, math.inf, -math.inf])
def test_validate_see_options_invalid_min_weight(invalid: float) -> None:
    with pytest.raises(ValueError, match="min_weight must be finite and nonnegative."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=invalid,
            edge_opacity=0.45,
        )


@pytest.mark.parametrize("invalid", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_validate_see_options_invalid_edge_opacity(invalid: float) -> None:
    with pytest.raises(ValueError, match="edge_opacity must be between 0 and 1."):
        validate_see_options(
            view="auto",
            theme="light",
            color_by="auto",
            detail="summary",
            max_neurons=100,
            max_connections=1500,
            sample=0,
            min_weight=0.0,
            edge_opacity=invalid,
        )


def test_validate_tensor_view_options_interactive_on_module() -> None:
    with pytest.raises(
        SeeDLError, match="only supported for PyTorch tensors and NumPy arrays, not nn.Module"
    ):
        validate_tensor_view_options(
            _DummyModule(),
            view="interactive",
            dim_names=None,
            max_frames=32,
            compare=None,
            is_tensor=False,
            is_multidim_tensor=False,
            is_module=True,
        )


def test_validate_tensor_view_options_interactive_on_scalar() -> None:
    with pytest.raises(
        SeeDLError, match="only supported for PyTorch tensors and NumPy arrays, got int"
    ):
        validate_tensor_view_options(
            42,
            view="explorer",
            dim_names=None,
            max_frames=32,
            compare=None,
            is_tensor=False,
            is_multidim_tensor=False,
            is_module=False,
        )


def test_validate_tensor_view_options_interactive_on_1d_tensor() -> None:
    dummy_1d = _DummyTensor1D()
    with pytest.raises(SeeDLError, match=r"requires at least 2 dimensions, got shape \(10,\)"):
        validate_tensor_view_options(
            dummy_1d,
            view="interactive",
            dim_names=None,
            max_frames=32,
            compare=None,
            is_tensor=True,
            is_multidim_tensor=False,
            is_module=False,
        )


def test_validate_tensor_view_options_dim_names_on_module() -> None:
    with pytest.raises(
        SeeDLError,
        match="dim_names= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module",
    ):
        validate_tensor_view_options(
            _DummyModule(),
            view="auto",
            dim_names=["batch", "channels"],
            max_frames=32,
            compare=None,
            is_tensor=False,
            is_multidim_tensor=False,
            is_module=True,
        )


def test_validate_tensor_view_options_dim_names_on_unsupported_object() -> None:
    with pytest.raises(
        SeeDLError,
        match="dim_names= argument is only supported for PyTorch tensors and NumPy arrays, got list",
    ):
        validate_tensor_view_options(
            [1, 2, 3],
            view="auto",
            dim_names=["items"],
            max_frames=32,
            compare=None,
            is_tensor=False,
            is_multidim_tensor=False,
            is_module=False,
        )


@pytest.mark.parametrize("invalid_frames", [0, -1, 1.5, "10", True, False])
def test_validate_tensor_view_options_max_frames_invalid(invalid_frames: Any) -> None:
    with pytest.raises(ValueError, match="max_frames must be a positive integer."):
        validate_tensor_view_options(
            _DummyTensor2D(),
            view="auto",
            dim_names=None,
            max_frames=invalid_frames,
            compare=None,
            is_tensor=True,
            is_multidim_tensor=True,
            is_module=False,
        )


def test_validate_tensor_view_options_explicit_view_with_compare() -> None:
    with pytest.raises(ValueError, match="Explicit view options require a model without compare=."):
        validate_tensor_view_options(
            _DummyModule(),
            view="neurons",
            dim_names=None,
            max_frames=32,
            compare=_DummyModule(),
            is_tensor=False,
            is_multidim_tensor=False,
            is_module=True,
        )


def test_resolve_see_backend_3d_requires_interactive() -> None:
    with pytest.raises(ValueError, match="view='3d' requires Plotly and interactive rendering."):
        resolve_see_backend("auto", view="3d", interactive=False, is_multidim_tensor=False)


def test_resolve_see_backend_interactive_type_error() -> None:
    with pytest.raises(TypeError, match="interactive must be True, False, or None."):
        resolve_see_backend("auto", view="auto", interactive="yes", is_multidim_tensor=False)  # type: ignore[arg-type]


def test_resolve_see_backend_interactive_tensor_auto_is_svg() -> None:
    backend = resolve_see_backend(
        "auto", view="interactive", interactive=True, is_multidim_tensor=True
    )
    assert backend == "svg"


def test_resolve_see_backend_interactive_false_auto_is_svg() -> None:
    backend = resolve_see_backend("auto", view="auto", interactive=False, is_multidim_tensor=False)
    assert backend == "svg"


def test_validate_object_arguments_input_on_non_module() -> None:
    with pytest.raises(
        SeeDLError, match="The input= argument is only used when visualizing a PyTorch nn.Module."
    ):
        validate_object_arguments(
            [1, 2, 3],
            input=[1],
            slice=None,
            layout=None,
            is_module=False,
            is_tensor=False,
        )


def test_validate_object_arguments_slice_on_module() -> None:
    with pytest.raises(
        SeeDLError,
        match="The slice= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module",
    ):
        validate_object_arguments(
            _DummyModule(),
            input=None,
            slice=0,
            layout=None,
            is_module=True,
            is_tensor=False,
        )


def test_validate_object_arguments_layout_on_module() -> None:
    with pytest.raises(
        SeeDLError,
        match="The layout= argument is only supported for PyTorch tensors and NumPy arrays, not nn.Module",
    ):
        validate_object_arguments(
            _DummyModule(),
            input=None,
            slice=None,
            layout="CHW",
            is_module=True,
            is_tensor=False,
        )


def test_validate_object_arguments_layout_on_non_tensor() -> None:
    with pytest.raises(
        SeeDLError,
        match="The layout= argument is only supported for PyTorch tensors and NumPy arrays, got int",
    ):
        validate_object_arguments(
            42,
            input=None,
            slice=None,
            layout="CHW",
            is_module=False,
            is_tensor=False,
        )
