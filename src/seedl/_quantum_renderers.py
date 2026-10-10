"""Plotly and SVG renderers for quantum visualizations:
Bloch spheres, quantum circuits, statevectors, density matrices, and entanglement.
"""

from __future__ import annotations

import html
import math
from typing import Any

from ._quantum import (
    compute_bloch_vector,
    get_single_qubit_gate,
)
from .exceptions import MissingDependencyError

# Quantum Palette
Q_CYAN = "#00e5ff"
Q_AMBER = "#fbbf24"
Q_PURPLE = "#a855f7"
Q_TEAL = "#14b8a6"
Q_ROSE = "#f43f5e"
Q_BLUE = "#3b82f6"


def _quantum_theme(theme_name: str) -> tuple[str, str, str, str]:
    if theme_name in {"dark", "scientific"}:
        return ("#0f172a", "#f8fafc", "#94a3b8", "#1e293b")
    if theme_name == "minimal":
        return ("#ffffff", "#0f172a", "#64748b", "#e2e8f0")
    if theme_name == "presentation":
        return ("#050814", "#ffffff", "#cbd5e1", "#1e293b")
    # light
    return ("#f8fafc", "#0f172a", "#64748b", "#e2e8f0")


# =========================================================================
# 1. 3D BLOCH SPHERE (PLOTLY)
# =========================================================================


def plotly_bloch(view: Any) -> Any:
    """Render interactive 3D Bloch sphere using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError(
            "Interactive Bloch sphere requires Plotly: `pip install 'seedl[plotly]'`."
        ) from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    fig = go.Figure()

    # 1. Wireframe sphere circles (Latitude and Longitude)
    num_pts = 64
    theta_pts = [2.0 * math.pi * i / num_pts for i in range(num_pts + 1)]

    # Equator (z = 0, radius = 1)
    eq_x = [math.cos(t) for t in theta_pts]
    eq_y = [math.sin(t) for t in theta_pts]
    eq_z = [0.0 for _ in theta_pts]

    fig.add_trace(
        go.Scatter3d(
            x=eq_x,
            y=eq_y,
            z=eq_z,
            mode="lines",
            line=dict(color=Q_CYAN, width=3, dash="dot"),
            name="Equator (z=0)",
            hoverinfo="skip",
            opacity=0.6,
        )
    )

    # Parallels (z = 0.5 and z = -0.5)
    r_half = math.sqrt(1.0 - 0.5**2)
    for z_level, _sign_name in [(0.5, "z=+0.5"), (-0.5, "z=-0.5")]:
        px = [r_half * math.cos(t) for t in theta_pts]
        py = [r_half * math.sin(t) for t in theta_pts]
        pz = [z_level for _ in theta_pts]
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="lines",
                line=dict(color=border, width=1.5),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.35,
            )
        )

    # Prime meridian (xz-plane: y = 0)
    fig.add_trace(
        go.Scatter3d(
            x=[math.sin(t) for t in theta_pts],
            y=[0.0 for _ in theta_pts],
            z=[math.cos(t) for t in theta_pts],
            mode="lines",
            line=dict(color=border, width=1.5),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.35,
        )
    )

    # Cross meridian (yz-plane: x = 0)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0 for _ in theta_pts],
            y=[math.sin(t) for t in theta_pts],
            z=[math.cos(t) for t in theta_pts],
            mode="lines",
            line=dict(color=border, width=1.5),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.35,
        )
    )

    # Translucent sphere surface
    u_vals = [math.pi * i / 24 for i in range(25)]
    v_vals = [2.0 * math.pi * j / 24 for j in range(25)]
    surf_x = [[math.sin(u) * math.cos(v) for v in v_vals] for u in u_vals]
    surf_y = [[math.sin(u) * math.sin(v) for v in v_vals] for u in u_vals]
    surf_z = [[math.cos(u) for _ in v_vals] for u in u_vals]

    fig.add_trace(
        go.Surface(
            x=surf_x,
            y=surf_y,
            z=surf_z,
            opacity=0.08,
            colorscale=[[0, "#38bdf8"], [1, "#818cf8"]],
            showscale=False,
            hoverinfo="skip",
            name="Bloch Sphere Surface",
        )
    )

    # Coordinate Axes: X, Y, Z
    axis_lim = 1.3
    axis_lines = [
        ("X", [-axis_lim, axis_lim], [0, 0], [0, 0]),
        ("Y", [0, 0], [-axis_lim, axis_lim], [0, 0]),
        ("Z", [0, 0], [0, 0], [-axis_lim, axis_lim]),
    ]
    for _name, ax, ay, az in axis_lines:
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="lines",
                line=dict(color=muted, width=2.5),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.7,
            )
        )

    # Cardinal Basis Labels: |0>, |1>, |+>, |->, |+i>, |-i>
    cardinal_pts = [
        (0.0, 0.0, 1.15, "<b>|0⟩</b><br>(North Pole)", "+Z"),
        (0.0, 0.0, -1.15, "<b>|1⟩</b><br>(South Pole)", "-Z"),
        (1.18, 0.0, 0.0, "<b>|+⟩</b>", "+X"),
        (-1.18, 0.0, 0.0, "<b>|-⟩</b>", "-X"),
        (0.0, 1.18, 0.0, "<b>|+i⟩</b>", "+Y"),
        (0.0, -1.18, 0.0, "<b>|-i⟩</b>", "-Y"),
    ]
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] for p in cardinal_pts],
            y=[p[1] for p in cardinal_pts],
            z=[p[2] for p in cardinal_pts],
            mode="text",
            text=[p[3] for p in cardinal_pts],
            textfont=dict(color=fg, size=13),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Bloch vector line and marker
    bx, by, bz = data["x"], data["y"], data["z"]
    br = data["r"]
    purity = data["purity"]
    is_pure = data["is_pure"]
    prob_0 = data["prob_0"]
    prob_1 = data["prob_1"]

    vec_color = Q_AMBER if is_pure else "#ec4899"
    hover_card = (
        f"<b>Bloch Vector</b><br>"
        f"State: {view.title}<br>"
        f"x = {bx:+.4f}<br>"
        f"y = {by:+.4f}<br>"
        f"z = {bz:+.4f}<br>"
        f"|r| = {br:.4f} ({'Pure' if is_pure else 'Mixed'})<br>"
        f"Purity = {purity:.4f}<br>"
        f"P(|0⟩) = {prob_0:.1%}<br>"
        f"P(|1⟩) = {prob_1:.1%}<br>"
        f"θ = {math.degrees(data['theta']):.1f}°, φ = {math.degrees(data['phi']):.1f}°"
    )

    # Vector line from origin to (bx, by, bz)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, bx],
            y=[0.0, by],
            z=[0.0, bz],
            mode="lines",
            line=dict(color=vec_color, width=6),
            name="Bloch Vector",
            hoverinfo="text",
            text=[hover_card, hover_card],
            opacity=1.0,
        )
    )

    # Vector endpoint marker
    fig.add_trace(
        go.Scatter3d(
            x=[bx],
            y=[by],
            z=[bz],
            mode="markers",
            marker=dict(size=10, color=vec_color, line=dict(color="#ffffff", width=2)),
            name="State Vector Tip",
            hoverinfo="text",
            text=[hover_card],
        )
    )

    # Detail annotation panel
    detail_html = (
        f"<b>QUANTUM STATE: {html.escape(view.title)}</b><br>"
        f"<b>Coordinates:</b> x = {bx:+.4f}, y = {by:+.4f}, z = {bz:+.4f} · "
        f"<b>Length:</b> |r| = {br:.4f} ({'Pure state' if is_pure else 'Mixed state'})<br>"
        f"<b>Angles:</b> θ = {math.degrees(data['theta']):.1f}°, φ = {math.degrees(data['phi']):.1f}° · "
        f"<b>Purity:</b> Tr(ρ²) = {purity:.4f}<br>"
        f"<b>Probabilities:</b> P(|0⟩) = {prob_0:.1%} · P(|1⟩) = {prob_1:.1%}"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>Interactive 3D Bloch Sphere</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=22, color=fg),
        ),
        paper_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=720,
        margin=dict(l=30, r=30, t=90, b=110),
        annotations=[
            dict(
                x=0,
                y=-0.12,
                xref="paper",
                yref="paper",
                text=detail_html,
                showarrow=False,
                xanchor="left",
                align="left",
                font=dict(size=12, color=fg),
                bgcolor=bg,
                bordercolor=border,
                borderpad=8,
            )
        ],
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=bg,
            aspectmode="cube",
            camera=dict(
                eye=dict(x=-1.6, y=-1.8, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            dragmode="orbit",
        ),
        uirevision="seedl-bloch",
    )

    # Interactive Gate buttons (X, Y, Z, H, S, T)
    gate_buttons = []
    # Current state as initial
    gate_buttons.append(
        dict(
            label="Current State",
            method="restyle",
            args=[
                {
                    "x": [[0.0, bx], [bx]],
                    "y": [[0.0, by], [by]],
                    "z": [[0.0, bz], [bz]],
                },
                [len(fig.data) - 2, len(fig.data) - 1],
            ],
        )
    )

    # Precalculate gate states if pure
    if is_pure and data.get("amplitudes"):
        vec = [complex(a["real"], a["imag"]) for a in data["amplitudes"]]
        for gate_name in ["H", "X", "Y", "Z", "S", "T"]:
            try:
                g_mat = get_single_qubit_gate(gate_name)
                # out = g_mat * vec
                g_vec = [
                    g_mat[0][0] * vec[0] + g_mat[0][1] * vec[1],
                    g_mat[1][0] * vec[0] + g_mat[1][1] * vec[1],
                ]
                gx, gy, gz, gp = compute_bloch_vector(g_vec)
                gate_buttons.append(
                    dict(
                        label=f"Apply {gate_name}",
                        method="restyle",
                        args=[
                            {
                                "x": [[0.0, gx], [gx]],
                                "y": [[0.0, gy], [gy]],
                                "z": [[0.0, gz], [gz]],
                            },
                            [len(fig.data) - 2, len(fig.data) - 1],
                        ],
                    )
                )
            except Exception:
                pass

    if len(gate_buttons) > 1:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.04,
                    y=1.06,
                    xanchor="left",
                    yanchor="top",
                    bgcolor=bg,
                    bordercolor=border,
                    font=dict(size=12, color=fg),
                    buttons=gate_buttons,
                )
            ]
        )

    return fig


def svg_bloch(view: Any) -> str:
    """Render 2D isometric projected Bloch sphere for static SVG export."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    width, height = 780, 560
    cx, cy = 340, 270
    radius = 180.0

    # Isometric projection: x goes right-down, y goes right-up, z goes up
    # proj_x = cx + radius * (x * cos(30°) - y * cos(30°))
    # proj_y = cy - radius * z + radius * (x * sin(30°) + y * sin(30°)) * 0.4
    cos30 = math.cos(math.radians(30))
    sin30 = math.sin(math.radians(30))

    def project(x: float, y: float, z: float) -> tuple[float, float]:
        px = cx + radius * (x * cos30 - y * cos30)
        py = cy - radius * z + radius * (x * sin30 + y * sin30) * 0.4
        return (px, py)

    bx, by, bz = data["x"], data["y"], data["z"]
    tip_x, tip_y = project(bx, by, bz)

    # Projected poles
    north_x, north_y = project(0, 0, 1)
    south_x, south_y = project(0, 0, -1)
    plus_x, plus_y = project(1, 0, 0)
    minus_x, minus_y = project(-1, 0, 0)
    plus_i_x, plus_i_y = project(0, 1, 0)
    minus_i_x, minus_i_y = project(0, -1, 0)

    # Projected axis tips
    ax_x, ax_y = project(1.25, 0, 0)
    ay_x, ay_y = project(0, 1.25, 0)
    az_x, az_y = project(0, 0, 1.25)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        # Header
        f'<text x="35" y="45" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / QUANTUM COMPUTING</text>',
        f'<text x="35" y="80" font-family="system-ui, sans-serif" font-size="24" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="35" y="105" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">Interactive 3D Bloch Sphere · |r| = {data["r"]:.4f} · Purity = {data["purity"]:.4f}</text>',
        # Main Sphere Circle
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{border}" stroke-width="2"/>',
        # Equator Ellipse
        f'<ellipse cx="{cx}" cy="{cy}" rx="{radius}" ry="{radius * 0.4}" fill="none" stroke="{Q_CYAN}" stroke-width="1.8" stroke-dasharray="4,4" opacity="0.6"/>',
        # Z-axis
        f'<line x1="{cx}" y1="{cy + radius * 1.2}" x2="{cx}" y2="{cy - radius * 1.25}" stroke="{muted}" stroke-width="1.5" stroke-dasharray="3,3"/>',
        # X-axis
        f'<line x1="{cx - radius * cos30 * 1.2}" y1="{cy - radius * sin30 * 0.4 * 1.2}" x2="{ax_x}" y2="{ax_y}" stroke="{muted}" stroke-width="1.5"/>',
        # Y-axis
        f'<line x1="{cx + radius * cos30 * 1.2}" y1="{cy - radius * sin30 * 0.4 * 1.2}" x2="{ay_x}" y2="{ay_y}" stroke="{muted}" stroke-width="1.5"/>',
        # Axis Labels
        f'<text x="{az_x + 8}" y="{az_y + 4}" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="{fg}">+Z (|0⟩)</text>',
        f'<text x="{cx + 8}" y="{cy + radius + 18}" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="{fg}">-Z (|1⟩)</text>',
        f'<text x="{ax_x + 6}" y="{ax_y + 4}" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{muted}">+X (|+⟩)</text>',
        f'<text x="{ay_x + 6}" y="{ay_y + 4}" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{muted}">+Y (|+i⟩)</text>',
        # Poles
        f'<circle cx="{north_x}" cy="{north_y}" r="3.5" fill="{muted}"/>',
        f'<circle cx="{south_x}" cy="{south_y}" r="3.5" fill="{muted}"/>',
        f'<circle cx="{plus_x}" cy="{plus_y}" r="3" fill="{muted}"/>',
        f'<circle cx="{minus_x}" cy="{minus_y}" r="3" fill="{muted}"/>',
        f'<circle cx="{plus_i_x}" cy="{plus_i_y}" r="3" fill="{muted}"/>',
        f'<circle cx="{minus_i_x}" cy="{minus_i_y}" r="3" fill="{muted}"/>',
        # Bloch Vector Arrow
        f'<line x1="{cx}" y1="{cy}" x2="{tip_x}" y2="{tip_y}" stroke="{Q_AMBER}" stroke-width="4.5" stroke-linecap="round"/>',
        f'<circle cx="{tip_x}" cy="{tip_y}" r="7" fill="{Q_AMBER}" stroke="#ffffff" stroke-width="2"/>',
        # Stats Card Sidebar (right side)
        f'<rect x="560" y="140" width="190" height="260" rx="8" fill="{border}" opacity="0.4"/>',
        f'<text x="575" y="170" font-family="system-ui, sans-serif" font-size="14" font-weight="700" fill="{fg}">State Data</text>',
        f'<text x="575" y="200" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">x: <tspan fill="{fg}" font-weight="600">{bx:+.4f}</tspan></text>',
        f'<text x="575" y="224" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">y: <tspan fill="{fg}" font-weight="600">{by:+.4f}</tspan></text>',
        f'<text x="575" y="248" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">z: <tspan fill="{fg}" font-weight="600">{bz:+.4f}</tspan></text>',
        f'<text x="575" y="278" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">|r|: <tspan fill="{fg}" font-weight="600">{data["r"]:.4f}</tspan></text>',
        f'<text x="575" y="302" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">Purity: <tspan fill="{fg}" font-weight="600">{data["purity"]:.4f}</tspan></text>',
        f'<text x="575" y="332" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">P(|0⟩): <tspan fill="{fg}" font-weight="600">{data["prob_0"]:.1%}</tspan></text>',
        f'<text x="575" y="356" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">P(|1⟩): <tspan fill="{fg}" font-weight="600">{data["prob_1"]:.1%}</tspan></text>',
        f'<text x="575" y="386" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">θ: <tspan fill="{fg}">{math.degrees(data["theta"]):.1f}°</tspan>, φ: <tspan fill="{fg}">{math.degrees(data["phi"]):.1f}°</tspan></text>',
        "</svg>",
    ]
    return "\n".join(svg_lines)


# =========================================================================
# 2. QUANTUM CIRCUIT VISUALIZATION
# =========================================================================


def plotly_circuit(view: Any) -> Any:
    """Render interactive quantum circuit diagram using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError(
            "Circuit visualization requires Plotly: `pip install 'seedl[plotly]'`."
        ) from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    num_qubits = data["num_qubits"]
    instructions = data["instructions"]
    num_steps = max(len(instructions), 1)

    fig = go.Figure()

    # Wire layout: Y-axis has qubits 0 to num_qubits - 1 (qubit 0 at top)
    wire_y = [num_qubits - 1 - i for i in range(num_qubits)]

    # Draw horizontal qubit wires
    for q in range(num_qubits):
        y = wire_y[q]
        fig.add_trace(
            go.Scatter(
                x=[-0.5, num_steps + 0.5],
                y=[y, y],
                mode="lines",
                line=dict(color=border, width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Qubit label
        fig.add_annotation(
            x=-0.6,
            y=y,
            text=f"<b>q_{q}</b>",
            showarrow=False,
            xanchor="right",
            font=dict(size=14, color=fg),
        )

    # Draw operations
    gate_colors = {
        "H": Q_CYAN,
        "X": Q_ROSE,
        "Y": Q_PURPLE,
        "Z": Q_AMBER,
        "S": Q_TEAL,
        "T": "#06b6d4",
        "RX": "#f97316",
        "RY": "#ec4899",
        "RZ": "#8b5cf6",
        "CX": Q_CYAN,
        "CZ": Q_AMBER,
        "MEASURE": "#64748b",
        "BARRIER": "#475569",
    }

    for step_idx, inst in enumerate(instructions):
        x = step_idx + 0.5
        name = inst["name"].upper()
        target_qs = inst["qubits"]

        if name == "BARRIER":
            # Vertical dashed line across all wires
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[wire_y[num_qubits - 1] - 0.4, wire_y[0] + 0.4],
                    mode="lines",
                    line=dict(color=muted, width=2, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            continue

        if name in {"CX", "CNOT"} and len(target_qs) == 2:
            ctrl_q, tgt_q = target_qs[0], target_qs[1]
            cy = wire_y[ctrl_q]
            ty = wire_y[tgt_q]

            # Connecting vertical line
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[cy, ty],
                    mode="lines",
                    line=dict(color=Q_CYAN, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Control dot
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[cy],
                    mode="markers",
                    marker=dict(size=14, color=Q_CYAN),
                    name="Control",
                    hoverinfo="text",
                    text=[f"Control on q_{ctrl_q}"],
                    showlegend=False,
                )
            )
            # Target target symbol (oplus)
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[ty],
                    mode="markers+text",
                    marker=dict(size=24, color=Q_CYAN, line=dict(color="#ffffff", width=2)),
                    text=["⊕"],
                    textfont=dict(size=18, color="#0f172a"),
                    name="Target",
                    hoverinfo="text",
                    textposition="middle center",
                    showlegend=False,
                )
            )
            continue

        # Single or multi-qubit gate box
        color = gate_colors.get(name, Q_TEAL)
        for q in target_qs:
            y = wire_y[q]
            hover_text = (
                f"<b>Gate: {name}</b><br>"
                f"Target: q_{q}<br>"
                f"Parameters: {inst['params']}<br>"
                f"Step: {step_idx + 1}"
            )
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        symbol="square",
                        size=34,
                        color=color,
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    text=[name],
                    textfont=dict(
                        size=12, color="#ffffff" if name not in {"H", "Z"} else "#0f172a"
                    ),
                    textposition="middle center",
                    hoverinfo="text",
                    hovertext=[hover_text],
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>Quantum Circuit · Depth: {data['depth']} · Operations: {data['num_ops']}</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=22, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=max(360, num_qubits * 90 + 180),
        xaxis=dict(
            visible=False,
            range=[-1.0, num_steps + 1.0],
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False,
            range=[-0.8, num_qubits - 0.2],
            fixedrange=True,
        ),
        margin=dict(l=70, r=40, t=100, b=50),
    )

    return fig


def svg_circuit(view: Any) -> str:
    """Render QuantumCircuit as SVG."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    num_qubits = data["num_qubits"]
    instructions = data["instructions"]
    num_steps = max(len(instructions), 1)

    left_margin = 85
    step_width = 65
    width = max(680, left_margin + (num_steps + 1) * step_width)
    height = max(260, num_qubits * 75 + 160)

    wire_y = {q: 140 + q * 65 for q in range(num_qubits)}

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        # Header
        f'<text x="35" y="45" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / QUANTUM CIRCUIT</text>',
        f'<text x="35" y="80" font-family="system-ui, sans-serif" font-size="24" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="35" y="105" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">{num_qubits} Qubits · Depth {data["depth"]} · {data["num_ops"]} Operations</text>',
    ]

    # Draw horizontal wires
    for q in range(num_qubits):
        y = wire_y[q]
        svg_lines.append(
            f'<text x="{left_margin - 15}" y="{y + 5}" font-family="system-ui, monospace" font-size="14" font-weight="700" fill="{fg}" text-anchor="end">q_{q}</text>'
        )
        svg_lines.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{width - 40}" y2="{y}" stroke="{border}" stroke-width="2.5"/>'
        )

    # Draw gates
    for step_idx, inst in enumerate(instructions):
        x = left_margin + (step_idx + 1) * step_width
        name = inst["name"].upper()
        target_qs = inst["qubits"]

        if name == "BARRIER":
            svg_lines.append(
                f'<line x1="{x}" y1="{wire_y[0] - 25}" x2="{x}" y2="{wire_y[num_qubits - 1] + 25}" stroke="{muted}" stroke-width="2" stroke-dasharray="4,4"/>'
            )
            continue

        if name in {"CX", "CNOT"} and len(target_qs) == 2:
            cq, tq = target_qs[0], target_qs[1]
            cy, ty = wire_y[cq], wire_y[tq]
            # Line
            svg_lines.append(
                f'<line x1="{x}" y1="{cy}" x2="{x}" y2="{ty}" stroke="{Q_CYAN}" stroke-width="3"/>'
            )
            # Control dot
            svg_lines.append(f'<circle cx="{x}" cy="{cy}" r="6" fill="{Q_CYAN}"/>')
            # Target plus circle
            svg_lines.append(
                f'<circle cx="{x}" cy="{ty}" r="12" fill="{bg}" stroke="{Q_CYAN}" stroke-width="2.5"/>'
            )
            svg_lines.append(
                f'<line x1="{x - 8}" y1="{ty}" x2="{x + 8}" y2="{ty}" stroke="{Q_CYAN}" stroke-width="2.5"/>'
            )
            svg_lines.append(
                f'<line x1="{x}" y1="{ty - 8}" x2="{x}" y2="{ty + 8}" stroke="{Q_CYAN}" stroke-width="2.5"/>'
            )
            continue

        box_color = (
            Q_CYAN if name == "H" else Q_ROSE if name == "X" else Q_AMBER if name == "Z" else Q_TEAL
        )
        for q in target_qs:
            y = wire_y[q]
            svg_lines.append(
                f'<rect x="{x - 18}" y="{y - 18}" width="36" height="36" rx="4" fill="{box_color}" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(inst["label"])}</title></rect>'
            )
            txt_color = "#0f172a" if name in {"H", "Z"} else "#ffffff"
            svg_lines.append(
                f'<text x="{x}" y="{y + 5}" font-family="system-ui, sans-serif" font-size="12" font-weight="700" fill="{txt_color}" text-anchor="middle">{name}</text>'
            )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


# =========================================================================
# 3. STATEVECTOR & PROBABILITY VISUALIZATION
# =========================================================================


def plotly_statevector(view: Any) -> Any:
    """Render quantum statevector probabilities and phase distribution."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError(
            "Statevector visualization requires Plotly: `pip install 'seedl[plotly]'`."
        ) from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    basis_states = data["basis_states"]
    probs = data["probabilities"]
    phases = data["phases"]
    amplitudes = data["amplitudes"]

    # Color bars by phase in radians [-pi, pi] mapped to a continuous color scale
    fig = go.Figure()

    hover_texts = [
        f"<b>Basis State: |{bs}⟩</b><br>"
        f"Probability: {probs[i]:.2%}<br>"
        f"Amplitude: {amplitudes[i]['real']:+.4f} + {amplitudes[i]['imag']:+.4f}i<br>"
        f"Magnitude: {amplitudes[i]['mag']:.4f}<br>"
        f"Phase: {math.degrees(phases[i]):.1f}°"
        for i, bs in enumerate(basis_states)
    ]

    fig.add_trace(
        go.Bar(
            x=[f"|{bs}⟩" for bs in basis_states],
            y=probs,
            text=[f"{p:.1%}" if p >= 0.03 else "" for p in probs],
            textposition="outside",
            marker=dict(
                color=phases,
                colorscale="Viridis",
                colorbar=dict(
                    title="Phase (rad)",
                    tickvals=[-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                    ticktext=["-π", "-π/2", "0", "+π/2", "+π"],
                    thickness=12,
                    len=0.6,
                ),
                line=dict(color=border, width=1),
            ),
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>Quantum Statevector Probabilities and Phases</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=22, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=520,
        xaxis=dict(title="Computational Basis State", color=fg, gridcolor=border),
        yaxis=dict(
            title="Measurement Probability P(i) = |c_i|²",
            range=[0, 1.08],
            color=fg,
            gridcolor=border,
        ),
        margin=dict(l=60, r=40, t=100, b=80),
    )
    return fig


def svg_statevector(view: Any) -> str:
    """Render statevector probabilities as static SVG."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    basis_states = data["basis_states"]
    probs = data["probabilities"]
    n = len(basis_states)

    width, height = 760, 480
    left, top = 80, 140
    chart_w = width - left - 60
    chart_h = height - top - 90

    bar_gap = chart_w / max(n, 1)
    bar_w = max(18.0, bar_gap * 0.6)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        # Header
        f'<text x="35" y="45" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / STATEVECTOR</text>',
        f'<text x="35" y="80" font-family="system-ui, sans-serif" font-size="24" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="35" y="105" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">{n} Computational Basis States · Normalized</text>',
        # Axis lines
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="{border}" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="{border}" stroke-width="2"/>',
    ]

    for i, (bs, p) in enumerate(zip(basis_states, probs, strict=True)):
        bx = left + i * bar_gap + (bar_gap - bar_w) / 2
        bh = chart_h * p
        by = top + chart_h - bh

        svg_lines.append(
            f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" rx="3" fill="{Q_CYAN}" opacity="0.9"><title>|{bs}⟩: {p:.2%}</title></rect>'
        )
        if p >= 0.03:
            svg_lines.append(
                f'<text x="{bx + bar_w / 2}" y="{by - 6}" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="{fg}" text-anchor="middle">{p:.1%}</text>'
            )
        svg_lines.append(
            f'<text x="{bx + bar_w / 2}" y="{top + chart_h + 20}" font-family="system-ui, monospace" font-size="12" font-weight="600" fill="{muted}" text-anchor="middle">|{bs}⟩</text>'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


# =========================================================================
# 4. ENTANGLEMENT & BELL STATE VISUALIZATION
# =========================================================================


def plotly_entanglement(view: Any) -> Any:
    """Render multi-qubit entanglement analysis (joint probabilities, reduced states, entropy)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError(
            "Entanglement visualization requires Plotly: `pip install 'seedl[plotly]'`."
        ) from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Joint Measurement Probabilities", "Subsystem Entanglement Entropy"],
    )

    # Subplot 1: Joint probabilities bar chart
    basis = data["basis_states"]
    probs = data["probabilities"]
    fig.add_trace(
        go.Bar(
            x=[f"|{b}⟩" for b in basis],
            y=probs,
            text=[f"{p:.1%}" if p > 0.01 else "" for p in probs],
            textposition="outside",
            marker=dict(color=Q_CYAN, line=dict(color=border, width=1)),
            name="Joint Probabilities",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Entropy bar indicators for Qubit 0 and Qubit 1
    entropy_A = data["entropy_q0"]
    entropy_B = data["entropy_q1"]
    fig.add_trace(
        go.Bar(
            x=["Qubit 0 (S_A)", "Qubit 1 (S_B)"],
            y=[entropy_A, entropy_B],
            text=[f"{entropy_A:.3f} bit", f"{entropy_B:.3f} bit"],
            textposition="outside",
            marker=dict(color=[Q_AMBER, Q_PURPLE], line=dict(color=border, width=1)),
            name="von Neumann Entropy",
        ),
        row=1,
        col=2,
    )

    # Detail annotation
    ent_note = (
        f"<b>ENTANGLEMENT DIAGNOSTIC</b><br>"
        f"State: {view.title}<br>"
        f"Entropy S(ρ_A) = {entropy_A:.3f} bit · S(ρ_B) = {entropy_B:.3f} bit (Max = 1.0 bit)<br>"
        f"Reduced Bloch vector Qubit 0: ({data['bloch_q0'][0]:.2f}, {data['bloch_q0'][1]:.2f}, {data['bloch_q0'][2]:.2f}) · |r| = {math.sqrt(sum(c**2 for c in data['bloch_q0'][:3])):.2f}<br>"
        f"Reduced Bloch vector Qubit 1: ({data['bloch_q1'][0]:.2f}, {data['bloch_q1'][1]:.2f}, {data['bloch_q1'][2]:.2f}) · |r| = {math.sqrt(sum(c**2 for c in data['bloch_q1'][:3])):.2f}"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>Multi-Qubit Entanglement & Reduced Density States</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=22, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=540,
        showlegend=False,
        annotations=list(fig.layout.annotations or [])
        + [
            dict(
                x=0,
                y=-0.16,
                xref="paper",
                yref="paper",
                text=ent_note,
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color=fg),
                bgcolor=bg,
                bordercolor=border,
                borderpad=8,
            )
        ],
        margin=dict(l=50, r=40, t=100, b=100),
    )
    return fig


def svg_entanglement(view: Any) -> str:
    """Render entanglement analysis as static SVG."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg, fg, muted, border = _quantum_theme(theme)

    width, height = 780, 480
    entropy_A = data["entropy_q0"]
    entropy_B = data["entropy_q1"]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        # Header
        f'<text x="35" y="45" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / ENTANGLEMENT</text>',
        f'<text x="35" y="80" font-family="system-ui, sans-serif" font-size="24" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="35" y="105" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">Reduced von Neumann Entropy S(ρ_A) = {entropy_A:.3f} bit · S(ρ_B) = {entropy_B:.3f} bit</text>',
        # Card 1: Probabilities
        f'<rect x="35" y="130" width="360" height="280" rx="8" fill="{border}" opacity="0.3"/>',
        f'<text x="55" y="160" font-family="system-ui, sans-serif" font-size="15" font-weight="700" fill="{fg}">Joint Probabilities</text>',
    ]

    for i, (bs, p) in enumerate(zip(data["basis_states"], data["probabilities"], strict=True)):
        py = 195 + i * 48
        svg_lines.append(
            f'<text x="55" y="{py + 14}" font-family="system-ui, monospace" font-size="13" font-weight="600" fill="{fg}">|{bs}⟩</text>'
        )
        bar_len = 220 * p
        svg_lines.append(
            f'<rect x="110" y="{py}" width="{bar_len}" height="20" rx="3" fill="{Q_CYAN}"/>'
        )
        svg_lines.append(
            f'<text x="{120 + bar_len}" y="{py + 15}" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="{muted}">{p:.1%}</text>'
        )

    # Card 2: Entanglement Entropy Gauges
    svg_lines += [
        f'<rect x="420" y="130" width="325" height="280" rx="8" fill="{border}" opacity="0.3"/>',
        f'<text x="440" y="160" font-family="system-ui, sans-serif" font-size="15" font-weight="700" fill="{fg}">Entanglement Entropy</text>',
        f'<text x="440" y="185" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">S(ρ) = -Tr(ρ log₂ ρ) in bits (Max = 1.0 bit)</text>',
        # Gauge Qubit 0
        f'<text x="440" y="225" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="{fg}">Qubit 0 Subsystem: {entropy_A:.3f} bit</text>',
        f'<rect x="440" y="235" width="260" height="18" rx="4" fill="{border}"/>',
        f'<rect x="440" y="235" width="{260 * min(1.0, entropy_A)}" height="18" rx="4" fill="{Q_AMBER}"/>',
        # Gauge Qubit 1
        f'<text x="440" y="295" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="{fg}">Qubit 1 Subsystem: {entropy_B:.3f} bit</text>',
        f'<rect x="440" y="305" width="260" height="18" rx="4" fill="{border}"/>',
        f'<rect x="440" y="305" width="{260 * min(1.0, entropy_B)}" height="18" rx="4" fill="{Q_PURPLE}"/>',
        # Summary
        f'<text x="440" y="365" font-family="system-ui, sans-serif" font-size="12" fill="{muted}">{"Maximally Entangled Bell Pair (1.0 bit)" if abs(entropy_A - 1.0) < 1e-4 else "Separable / Product State (0.0 bit)" if entropy_A < 1e-4 else "Partially Entangled State"}</text>',
        "</svg>",
    ]
    return "\n".join(svg_lines)
