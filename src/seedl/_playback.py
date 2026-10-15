"""Kernel-free playback shared by recorded training and quantum circuit evolution."""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any

from .exceptions import SeeDLError
from .view import Visualization

MAX_EXPORT_BYTES = 32 * 1024 * 1024


def training_view(tracker: Any, *, backend: str, color_by: str) -> Visualization:
    from .renderers import resolve_backend

    if color_by not in {"activation", "gradient"}:
        raise ValueError("color_by must be activation or gradient.")
    if not tracker.snapshots:
        raise SeeDLError(
            "No training snapshots recorded. Call capture() inside the tracker context."
        )
    weight_scale = color_scale = 0.0
    for snapshot in tracker.snapshots:
        graph = snapshot.data["graph"]
        if graph:
            weight_scale = max(
                weight_scale, max((abs(e["weight"]) for e in graph["edges"]), default=0)
            )
            color_scale = max(
                color_scale,
                max(
                    (abs(n[color_by]) for n in graph["nodes"] if n[color_by] is not None), default=0
                ),
            )
    return Visualization(
        kind="training_playback",
        title=f"{tracker.name} · recorded training",
        summary=f"{len(tracker.snapshots)} retained training snapshots",
        data={
            "snapshots": tracker.snapshots,
            "color_by": color_by,
            "weight_scale": weight_scale or 1,
            "color_scale": color_scale or 1,
        },
        metadata=tracker.recording_stats,
        notes=[
            "Only recorded passes are shown; no interpolation. Capture before optimizer.step().",
            "Activations are from the user's forward pass; gradients are those present at capture.",
            "Parameter/output previews are deterministic samples; missing or non-finite values remain unavailable.",
            "SVG/Matplotlib show the first retained snapshot. HTML playback requires Plotly.",
        ],
        backend=resolve_backend(backend),
    )


def training_figure(view: Visualization, index: int = 0) -> Any:
    import plotly.graph_objects as go

    from ._network_renderers import plotly_network
    from ._recording import restore_graph

    snapshot = view.data["snapshots"][index].data
    graph = restore_graph(snapshot, view.data["color_by"])
    if graph:
        graph.update(weight_scale=view.data["weight_scale"], color_scale=view.data["color_scale"])
        figure = plotly_network(
            Visualization("model", view.title, view.summary, {"neuron_graph": graph})
        )
        # Inspection is handled by the shared playback panel; avoid duplicated menu payloads.
        figure.update_layout(updatemenus=[])
    else:
        figure = go.Figure()
        gradient = view.data["color_by"] == "gradient"
        for name, parameter in snapshot["parameters"].items():
            values = parameter["gradients"] if gradient else parameter["values"]
            figure.add_trace(
                go.Scatter(
                    x=parameter["indices"],
                    y=values if values is not None else [None] * len(parameter["indices"]),
                    mode="markers",
                    name=name,
                )
            )
        figure.update_layout(
            xaxis_title="Original flat parameter index",
            yaxis_title="Gradient" if gradient else "Weight",
            height=600,
        )
    figure.update_layout(
        title_text=f"{html.escape(view.title)} · step {snapshot['step']} · epoch {snapshot['epoch']}",
        uirevision="seedl-training",
    )
    return figure


def quantum_pair(view: Visualization, index: int) -> tuple[Any, Any]:
    steps = view.data["evolution"]
    invalid = {
        "measurement_halt",
        "reset_halt",
        "conditional_halt",
        "post_halt",
        "simulation_error",
        "unsupported_halt",
    }
    halted_before = index > 0 and not steps[index - 1].get("statevector_valid", True)
    halted_now = any(steps[index].get(key) for key in invalid)
    before = None if halted_before else steps[max(0, index - 1)]
    after = None if halted_before or halted_now else steps[index]
    return before, after


def quantum_figure(view: Visualization, index: int = 0) -> Any:
    """Before/after reduced-qubit Bloch vectors and phase-colored basis probabilities."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from ._quantum_renderers import _quantum_theme

    before, after = quantum_pair(view, index)
    bg, fg, muted, border = _quantum_theme(view.data.get("theme", "scientific"))
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Before · reduced qubit states",
            "After · reduced qubit states",
            "Before · basis probabilities / phase",
            "After · basis probabilities / phase",
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.13,
    )
    theta = [2 * math.pi * i / 48 for i in range(49)]
    for col, state in enumerate((before, after), 1):
        if state is None:
            figure.add_annotation(
                x=0.24 if col == 1 else 0.76,
                y=0.65,
                xref="paper",
                yref="paper",
                text="State unavailable: evolution halted",
                showarrow=False,
            )
            continue
        for q, (bx, by, bz, _purity) in enumerate(state["bloch_vectors"]):
            center = q * 2.7
            xs: list[Any] = []
            ys: list[Any] = []
            zs: list[Any] = []
            for plane in range(3):
                for t in theta:
                    xs.append(center + (math.cos(t) if plane != 2 else 0))
                    ys.append(math.sin(t) if plane != 1 else 0)
                    zs.append(math.sin(t) if plane == 1 else math.cos(t) if plane == 2 else 0)
                xs.append(None)
                ys.append(None)
                zs.append(None)
            figure.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=muted, width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[center, center + bx],
                    y=[0, by],
                    z=[0, bz],
                    mode="lines+markers+text",
                    text=["", f"q{q}"],
                    textposition="top center",
                    line=dict(color="#0d9488", width=7),
                    marker=dict(size=4),
                    name=f"q{q}",
                    hovertext=f"q{q}: ({bx:.5g}, {by:.5g}, {bz:.5g})",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        amplitudes = state["amplitudes"][:64]
        figure.add_trace(
            go.Bar(
                x=[a["state"] for a in amplitudes],
                y=state["probabilities"][:64],
                marker=dict(
                    color=[a["phase"] for a in amplitudes],
                    cmin=-math.pi,
                    cmax=math.pi,
                    colorscale="HSV",
                    showscale=col == 2,
                    colorbar=dict(title="Phase (rad)", len=0.3, y=0.18),
                ),
                customdata=[[a["real"], a["imag"], a["phase"]] for a in amplitudes],
                hovertemplate="|%{x}⟩<br>P=%{y:.6g}<br>Re=%{customdata[0]:.6g}<br>Im=%{customdata[1]:.6g}<br>phase=%{customdata[2]:.6g} rad<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        figure.update_xaxes(type="category", title="Basis (Qiskit little-endian)", row=2, col=col)
        figure.update_yaxes(range=[0, 1.05], row=2, col=col)
    axis = dict(visible=False, range=[-1.2, 1.2])
    scene = dict(
        xaxis=dict(visible=False, range=[-1.2, (view.data["num_qubits"] - 1) * 2.7 + 1.2]),
        yaxis=axis,
        zaxis=axis,
        aspectmode="data",
        uirevision="seedl-quantum",
    )
    step = view.data["evolution"][index]
    figure.update_layout(
        title_text=f"{html.escape(view.title)} · {step['step']}: {html.escape(step['name'])}",
        height=760,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font_color=fg,
        scene=scene,
        scene2=scene,
        uirevision="seedl-quantum",
        margin=dict(t=100, b=50, l=45, r=70),
    )
    return figure


def playback_figure(view: Visualization) -> Any:
    return training_figure(view) if view.kind == "training_playback" else quantum_figure(view)


def static_training(view: Visualization, backend: str) -> Any:
    from ._recording import restore_graph
    from .renderers import render

    snapshot = view.data["snapshots"][0].data
    graph = restore_graph(snapshot, view.data["color_by"])
    if graph:
        return render(
            Visualization(
                "model",
                view.title,
                view.summary,
                {"neuron_graph": graph, "presentation": "neurons"},
                backend=backend,
            ),
            backend,
        )
    # Existing metric renderer remains a dependency-free fallback.
    from ._training import parse_training_history

    data, summary, notes = parse_training_history(
        {"loss": [snapshot["loss"] if snapshot["loss"] is not None else float("nan")]}
    )
    return render(
        Visualization("training", view.title, summary, data, notes=notes, backend=backend), backend
    )


def playback_html(
    view: Visualization, *, full_html: bool = True, include_plotlyjs: bool = True
) -> str:
    """One Plotly runtime and one renderer; controls operate entirely in the browser."""
    import plotly.io as pio

    training = view.kind == "training_playback"
    count = len(view.data["snapshots"] if training else view.data["evolution"])
    frames = []
    total = 0
    for i in range(count):
        fig = training_figure(view, i) if training else quantum_figure(view, i)
        # Keep the layout only once. Later frames carry traces and changing labels.
        figure = json.loads(pio.to_json(fig))
        layout = (
            figure["layout"]
            if i == 0
            else {k: figure["layout"][k] for k in ("title", "annotations") if k in figure["layout"]}
        )
        if training:
            snapshot = view.data["snapshots"][i].data
            graph = snapshot["graph"]
            from ._network_renderers import _node_text
            from ._recording import restore_graph

            graph = restore_graph(snapshot, view.data["color_by"])
            cards = (
                {n["id"]: f"Neuron {n['id']}\n" + _node_text(n, graph) for n in graph["nodes"]}
                if graph
                else {}
            )
            gradient_max = max(
                (
                    abs(value)
                    for parameter in snapshot["parameters"].values()
                    for value in (parameter["gradients"] or [])
                    if value is not None
                ),
                default=None,
            )
            parameter_count = sum(math.prod(p["shape"]) for p in snapshot["parameters"].values())
            retained_count = sum(len(p["values"]) for p in snapshot["parameters"].values())
            coverage = f"Parameter values retained: {retained_count}/{parameter_count}. "
            output = snapshot["output"]
            coverage += (
                f"Output values retained: {len(output['values'])}/{math.prod(output['shape'])}. "
                if output
                else "Output unavailable. "
            )
            coverage += (
                f"Connections retained: {len(graph['edges'])}/{graph['total_connections']}."
                if graph
                else "Neuron graph unavailable."
            )
            frame = {
                "label": f"Step {snapshot['step']} · epoch {snapshot['epoch']} · loss {snapshot['loss']} · sampled max |gradient| {gradient_max}",
                "epoch": snapshot["epoch"],
                "step": snapshot["step"],
                "loss": snapshot["loss"],
                "coverage": coverage,
                "cards": cards,
            }
        else:
            step = view.data["evolution"][i]
            frame = {
                "label": f"{step['step']}: {step['name']} · {step['description']}",
                "epoch": None,
                "step": step["step"],
                "cards": {},
                "coverage": f"Basis states displayed per valid pane: {min(64, 2 ** view.data['num_qubits'])}/{2 ** view.data['num_qubits']}. Full bounded state remains in view.data.",
            }
        frame.update(data=figure["data"], layout=layout)
        total += len(json.dumps(frame, separators=(",", ":")).encode())
        if total > MAX_EXPORT_BYTES:
            raise SeeDLError(
                "Playback payload exceeds 32 MiB. Retain fewer snapshots or reduce graph/circuit limits."
            )
        frames.append(frame)
    payload = (
        json.dumps(frames, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    # Plotly supplies the unique plot ID and scopes the controller to this view.
    circuit_payload = "null"
    if not training:
        from ._quantum_renderers import plotly_circuit

        circuit_payload = (
            pio.to_json(plotly_circuit(view))
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
    replacements = {
        "FRAMES": payload,
        "TRAINING": "true" if training else "false",
        "CIRCUIT": circuit_payload,
        "SUMMARY": json.dumps(
            f"{count} retained snapshots; {view.metadata.get('bytes', 0)} encoded bytes; {view.metadata.get('dropped', 0)} evicted."
            if training
            else "Recorded quantum evolution; invalid post-halt states are unavailable."
        ),
    }
    # Replace template tokens once; user labels must never be rescanned as tokens.
    script = re.sub(
        r"__(FRAMES|TRAINING|CIRCUIT|SUMMARY)__", lambda match: replacements[match[1]], _CONTROLLER
    )
    body = pio.to_html(
        playback_figure(view),
        full_html=False,
        include_plotlyjs=False,
        post_script=script,
        config={"responsive": True, "scrollZoom": True},
    )
    prefix = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(view.title)}</title></head><body style="margin:16px;font-family:system-ui">'
        if full_html
        else ""
    )
    suffix = "</body></html>" if full_html else ""
    # Enforce the cap on actual serialized content, including escaping, the
    # initial figure, circuit diagram and controls. Only the fixed runtime is excluded.
    if len((prefix + body + suffix).encode("utf-8")) > MAX_EXPORT_BYTES:
        raise SeeDLError(
            "Playback payload exceeds 32 MiB. Retain fewer snapshots or reduce graph/circuit limits."
        )
    if include_plotlyjs:
        from plotly.offline import get_plotlyjs

        # Multiple inline views reuse the loaded runtime while each output remains
        # independently displayable after a notebook reload or output clear.
        body = "<script>if (!window.Plotly) {" + get_plotlyjs() + "}</script>" + body
    return prefix + body + suffix


_CONTROLLER = r"""
(function() {
 const gd = document.getElementById('{plot_id}');
 const frames = __FRAMES__, training = __TRAINING__;
 const circuit = __CIRCUIT__, summary = __SUMMARY__;
 const panel = document.createElement('section');
 panel.className = 'seedl-playback'; panel.setAttribute('aria-label', 'Recorded playback');
 panel.style.cssText = 'padding:12px;border:1px solid #94a3b8;border-radius:8px;font:14px system-ui;color:#0f172a;background:#f8fafc';
 panel.innerHTML = '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center"><button data-action="rewind">Rewind</button><button data-action="previous">Previous</button><button data-action="play">Play</button><button data-action="pause">Pause</button><button data-action="next">Next</button><label>Speed <select data-control="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label><label>Recorded step <select data-control="step"></select></label><label data-epoch-label>Epoch <select data-control="epoch"></select></label><label data-neuron-label>Neuron <select data-control="neuron"><option value="">Overview</option></select></label></div><input data-control="timeline" type="range" min="0" step="1" value="0" aria-label="Training or gate timeline" style="width:100%;margin:12px 0"><output data-status aria-live="polite"></output><div data-loss style="height:160px"></div><pre data-inspection style="white-space:pre-wrap;max-height:220px;overflow:auto"></pre>';
 const recording = document.createElement('p'); recording.dataset.recording = ''; recording.textContent = summary;
 const coverage = document.createElement('p'); coverage.dataset.coverage = ''; coverage.textContent = frames[0].coverage;
 panel.prepend(recording); panel.appendChild(coverage);
 gd.parentNode.insertBefore(panel, gd);
 const get = name => panel.querySelector('[data-control="'+name+'"]');
 const status = panel.querySelector('[data-status]'), inspection = panel.querySelector('[data-inspection]');
 const timeline = get('timeline'), steps = get('step'), epochs = get('epoch'), neurons = get('neuron');
 const option = (select, text, value) => {const el = document.createElement('option'); el.textContent = text; el.value = value; select.appendChild(el);};
 timeline.max = frames.length - 1;
 frames.forEach((f, i) => option(steps, training ? 'Step '+f.step : f.label.split(' · ')[0], i));
 const epochValues = new Set(); frames.forEach((f, i) => {if (f.epoch !== null && !epochValues.has(f.epoch)) {option(epochs, f.epoch, i); epochValues.add(f.epoch);}});
 panel.querySelector('[data-epoch-label]').hidden = !epochValues.size;
 panel.querySelector('[data-neuron-label]').hidden = !training || !Object.keys(frames[0].cards).length;
 Object.keys(frames[0].cards).forEach(id => option(neurons, id, id));
 const lossDiv = panel.querySelector('[data-loss]');
 if (training) Plotly.newPlot(lossDiv, [{x:frames.map(f=>f.step), y:frames.map(f=>f.loss), mode:'lines+markers', name:'Recorded loss', connectgaps:false}, {x:[frames[0].step], y:[frames[0].loss], mode:'markers', marker:{size:12,color:'#dc2626'}, name:'Current'}], {height:160, margin:{t:15,b:35,l:65,r:20}, xaxis:{title:{text:'Recorded step'}}, yaxis:{title:{text:'Loss'}}, showlegend:false}, {responsive:true,displayModeBar:false});
 else lossDiv.hidden = true;
 let current=0, selected='', hovered=null, pointer=null, timer=null, playing=false, busy=false, queued=null;
 const clone = value => JSON.parse(JSON.stringify(value));
 const inspect = () => {inspection.textContent = selected && frames[current].cards[selected] ? frames[current].cards[selected] : (training ? 'Select a neuron or click a neuron marker to inspect this recorded pass.' : frames[current].label); neurons.value = selected;};
 const pause = () => {playing=false; clearTimeout(timer); timer=null;};
 const schedule = () => {clearTimeout(timer); if(playing) timer=setTimeout(async()=>{if(current>=frames.length-1){pause();return;} await render(current+1); schedule();}, 900 / Number(get('speed').value));};
 async function render(index) {
   index=Math.max(0,Math.min(frames.length-1,Number(index)));
   if(busy){queued=index; return;}
   busy=true;
   try {
     const f=frames[index], layout=Object.assign(clone(frames[0].layout),clone(f.layout));
     ['scene','scene2'].forEach(key=>{if(gd.layout[key] && gd.layout[key].camera && layout[key]) layout[key].camera=clone(gd.layout[key].camera);});
     // Zero-duration react: there are no invented intermediate network/quantum states.
     await Plotly.react(gd, clone(f.data), layout, {responsive:true,scrollZoom:true});
     current=index; hovered=null; timeline.value=index; steps.value=index;
     const epochIndex=frames.findIndex(v=>v.epoch===f.epoch); if(epochIndex>=0) epochs.value=epochIndex;
     status.textContent=f.label; coverage.textContent=f.coverage; inspect();
     if(training) await Plotly.restyle(lossDiv,{x:[[f.step]],y:[[f.loss]]},[1]);
     gd.dataset.seedlStep=String(f.step); gd.dataset.seedlIndex=String(index);
   } catch(error) {pause(); status.textContent='Playback error: '+error.message; throw error;}
   finally {busy=false; if(queued!==null){const next=queued; queued=null; await render(next);}}
 }
 panel.addEventListener('click', async event=>{const action=event.target.dataset.action; if(!action)return;
   if(action==='pause'){pause();return;} if(action==='play'){playing=true;schedule();return;}
   pause(); await render(action==='rewind'?0:current+(action==='next'?1:-1));
 });
 [timeline,steps,epochs].forEach(control=>control.addEventListener('input',()=>{pause();render(control.value);}));
 neurons.addEventListener('change',()=>{selected=neurons.value;inspect();});
 get('speed').addEventListener('change',schedule);
 gd.on('plotly_click',event=>{const point=event.points && event.points[0]; if(point && point.customdata && frames[current].cards[point.customdata]) {selected=point.customdata;inspect();}});
 // Scatter3d can emit hover without plotly_click on some WebGL builds.
 // Resolve a real pointer click from the current hover, avoiding private Plotly internals.
 gd.on('plotly_hover',event=>{const point=event.points && event.points[0]; hovered=point && frames[current].cards[point.customdata] ? point.customdata : null;});
 gd.on('plotly_unhover',()=>{hovered=null;});
 gd.addEventListener('pointerdown',event=>{pointer={x:event.clientX,y:event.clientY};});
 gd.addEventListener('click',event=>{if(hovered && pointer && Math.hypot(event.clientX-pointer.x,event.clientY-pointer.y)<5 && !busy) {selected=hovered;inspect();} pointer=null;});

 gd.seedlPlayback={seek:async index=>{pause();await render(index);},pause, get index(){return current;},get playing(){return playing;}};
 if(circuit) {
   const diagram=document.createElement('div'); diagram.dataset.seedlCircuit='true'; gd.parentNode.appendChild(diagram);
   Plotly.newPlot(diagram,circuit.data,circuit.layout,{responsive:true}).then(()=>{
     diagram.on('plotly_click',event=>{const point=event.points && event.points[0]; if(point && Number.isInteger(point.customdata)){pause();render(point.customdata);}});
   });
 }

 status.textContent=frames[0].label; gd.dataset.seedlStep=String(frames[0].step); gd.dataset.seedlIndex='0'; inspect();
})();
"""
