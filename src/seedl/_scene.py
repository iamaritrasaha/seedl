"""Reusable scene graph representation for SeeDL scientific visualizations.

Separates scientific data extraction from geometric scene layout and
interactive rendering backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SpatialPosition:
    """3D spatial coordinate."""

    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class SceneNode:
    """A visual node (neuron, qubit, gate, or data point) in a scene."""

    id: str
    layer: int
    index: int
    label: str
    position: SpatialPosition
    color: str = "#168f9c"
    size: float = 12.0
    opacity: float = 1.0
    data: dict[str, Any] = field(default_factory=dict)
    tooltip: str = ""
    incoming: list[str] = field(default_factory=list)
    outgoing: list[str] = field(default_factory=list)


@dataclass
class SceneConnection:
    """A directed visual edge (synapse, wire, or coupling) between nodes."""

    source: str
    target: str
    weight: float = 1.0
    color: str = "#8390a6"
    width: float = 1.0
    opacity: float = 0.5
    data: dict[str, Any] = field(default_factory=dict)
    tooltip: str = ""


@dataclass
class SceneLayer:
    """A logical grouping of nodes (neural layer, circuit time step, channel plane)."""

    index: int
    label: str
    node_ids: list[str] = field(default_factory=list)
    operation: str = ""
    module: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraView:
    """3D viewpoint configuration."""

    eye: dict[str, float] = field(default_factory=lambda: {"x": -1.8, "y": -2.4, "z": 0.85})
    center: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    up: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 1.0})


@dataclass
class AnimationStep:
    """A single stage in a forward pass or quantum circuit evolution."""

    name: str
    label: str
    active_layer: int
    highlight_nodes: list[str] = field(default_factory=list)
    highlight_edges: list[tuple[str, str]] = field(default_factory=list)
    stage_data: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Scene:
    """Scientific visualization scene containing geometry, styling, and metadata."""

    title: str
    subtitle: str = ""
    dimensions: int = 3
    theme: str = "scientific"
    nodes: dict[str, SceneNode] = field(default_factory=dict)
    connections: list[SceneConnection] = field(default_factory=list)
    layers: list[SceneLayer] = field(default_factory=list)
    steps: list[AnimationStep] = field(default_factory=list)
    camera: CameraView = field(default_factory=CameraView)
    selected_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: SceneNode) -> None:
        self.nodes[node.id] = node

    def add_connection(self, conn: SceneConnection) -> None:
        self.connections.append(conn)
        if conn.source in self.nodes:
            self.nodes[conn.source].outgoing.append(conn.target)
        if conn.target in self.nodes:
            self.nodes[conn.target].incoming.append(conn.source)

    def add_layer(self, layer: SceneLayer) -> None:
        self.layers.append(layer)
