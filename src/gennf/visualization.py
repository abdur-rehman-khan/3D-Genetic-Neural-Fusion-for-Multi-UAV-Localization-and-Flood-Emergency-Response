"""Plotly 3D flood scene + dashboard charts. Playback uses precomputed Plotly frames so rotating,
zooming and play/pause/scrub all run in the browser (no server round trip per frame)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .flood_environment import FloodEnvironment
from .rescue_allocation import VictimStatus
from .simulation import Simulation
from .uav_network import UAVStatus

UAV_COLORS = {UAVStatus.IDLE.value: "#38bdf8", UAVStatus.TRANSIT.value: "#f59e0b", UAVStatus.DELIVERING.value: "#22c55e",
              UAVStatus.RETURNING.value: "#a78bfa", UAVStatus.CHARGING.value: "#94a3b8"}
VICTIM_COLORS = {VictimStatus.UNDETECTED.value: "#64748b", VictimStatus.DETECTED.value: "#ef4444",
                 VictimStatus.LOCALIZED.value: "#fb923c", VictimStatus.ASSIGNED.value: "#facc15",
                 VictimStatus.ASSISTED.value: "#22c55e", VictimStatus.FAILED.value: "#111827"}


class MeshBuilder:
    def __init__(self): self.v: list = []; self.f: list = []; self.i: list = []
    def _add(self, verts, faces, inten=0.0):
        o = len(self.v); self.v += [list(p) for p in verts]; self.f += [[a + o, b + o, c + o] for a, b, c in faces]
        self.i += [inten] * len(verts)
    def box(self, x, y, z, w, d, h, inten=0.0):
        vs = [(x, y, z), (x + w, y, z), (x + w, y + d, z), (x, y + d, z), (x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h)]
        fs = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        self._add(vs, fs, inten)
    def pyramid(self, x, y, z, w, h):
        vs = [(x - w, y - w, z), (x + w, y - w, z), (x + w, y + w, z), (x - w, y + w, z), (x, y, z + h)]
        self._add(vs, [(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    def prism(self, cx, cy, z, r, h, n=28):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vs = [(cx + r * np.cos(t), cy + r * np.sin(t), z) for t in a] + [(cx + r * np.cos(t), cy + r * np.sin(t), z + h) for t in a]
        fs = []
        for k in range(n):
            k2 = (k + 1) % n; fs += [(k, k2, n + k2), (k, n + k2, n + k)]
        self._add(vs, fs)
    def fan(self, cx, cy, z, rx, ry, n=48):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vs = [(cx, cy, z)] + [(cx + rx * np.cos(t), cy + ry * np.sin(t), z) for t in a]
        self._add(vs, [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)])
    def quad(self, x0, y0, x1, y1, z):
        self._add([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2), (0, 2, 3)])
    def trace(self, color=None, opacity=1.0, name="", intensity=None, colorscale=None, hover=None):
        a = np.array(self.v) if self.v else np.zeros((0, 3)); f = np.array(self.f) if self.f else np.zeros((0, 3), int)
        kw = dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], opacity=opacity, name=name,
                  showlegend=False, flatshading=True, hoverinfo="skip" if hover is None else "text", hovertext=hover,
                  lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05))
        if intensity is not None:
            kw.update(intensity=np.array(self.i), colorscale=colorscale, showscale=False)
        else:
            kw["color"] = color
        return go.Mesh3d(**kw)


def static_traces(env: FloodEnvironment) -> list:
    s = env.size; tr = []
    g = MeshBuilder(); g.quad(0, 0, s, s, -0.3); tr.append(g.trace("#6f7f55", 1.0, "ground"))
    sev_col = {1: "#7dd3fc", 2: "#38bdf8", 3: "#0369a1"}
    for sev in (1, 2, 3):
        m = MeshBuilder()
        for z in env.flood_zones:
            if z.severity == sev: m.fan(z.cx, z.cy, 0.15 + 0.1 * sev, z.rx, z.ry)
        if m.v: tr.append(m.trace(sev_col[sev], 0.72, f"Flood severity {sev}"))
    m = MeshBuilder(); m.quad(env.river[0], 0, env.river[1], s, 0.45); tr.append(m.trace("#075985", 0.85, "River / deep flood"))
    dry, wet = MeshBuilder(), MeshBuilder()
    for r in env.roads:
        (wet if r.submerged else dry).quad(min(r.x0, r.x1) - (r.width / 2 if r.x0 == r.x1 else 0), min(r.y0, r.y1) - (r.width / 2 if r.y0 == r.y1 else 0),
                                           max(r.x0, r.x1) + (r.width / 2 if r.x0 == r.x1 else 0), max(r.y0, r.y1) + (r.width / 2 if r.y0 == r.y1 else 0), 0.55)
    tr.append(dry.trace("#4b5563", 1.0, "Road")); tr.append(wet.trace("#1e4a6b", 0.8, "Submerged road"))
    b = MeshBuilder(); x0, x1, by, bw, dz = env.bridge
    b.box(x0, by - bw / 2, dz - 0.6, x1 - x0, bw, 0.6); b.box(x0, by - bw / 2, dz, x1 - x0, 0.6, 1.0); b.box(x0, by + bw / 2 - 0.6, dz, x1 - x0, 0.6, 1.0)
    for px in np.linspace(x0 + 2, x1 - 2, 4): b.box(px - 0.8, by - 1.0, 0, 1.6, 2.0, dz - 0.6)
    tr.append(b.trace("#9ca3af", 1.0, "Bridge"))
    bl = MeshBuilder()
    for q in env.buildings: bl.box(q.x, q.y, 0, q.w, q.d, q.h, inten=q.h)
    tr.append(bl.trace(intensity=True, colorscale=[[0, "#d6d3d1"], [1, "#64748b"]], name="Buildings"))
    trunk, crown = MeshBuilder(), MeshBuilder()
    for x, y, h in env.trees:
        trunk.box(x - 0.4, y - 0.4, 0, 0.8, 0.8, h * 0.4); crown.pyramid(x, y, h * 0.35, h * 0.28, h * 0.8)
    tr.append(trunk.trace("#78350f", 1.0, "Tree trunks")); tr.append(crown.trace("#15803d", 1.0, "Trees"))
    hz = MeshBuilder()
    for h in env.hazards: hz.prism(h.cx, h.cy, 0, h.radius, h.height)
    tr.append(hz.trace("#dc2626", 0.22, "Restricted / hazardous"))
    tr.append(go.Scatter3d(x=[h.cx for h in env.hazards], y=[h.cy for h in env.hazards], z=[h.height + 3 for h in env.hazards],
                           mode="text", text=[f"⚠ {h.label}" for h in env.hazards], textfont=dict(color="#fca5a5", size=11),
                           showlegend=False, hoverinfo="skip"))
    bs = MeshBuilder(); bx, by_ = env.base[0], env.base[1]
    bs.box(bx - 9, by_ - 9, 0, 18, 18, env.base_platform_h); bs.box(bx - 1.2, by_ - 1.2, env.base_platform_h, 2.4, 2.4, 12)
    tr.append(bs.trace("#f8fafc", 1.0, "Rescue base"))
    tr.append(go.Scatter3d(x=[bx], y=[by_], z=[env.base_platform_h + 17], mode="markers+text", text=["Rescue Base"],
                           textposition="top center", marker=dict(size=7, color="#ffffff", symbol="square", line=dict(color="#0ea5e9", width=3)),
                           textfont=dict(color="#ffffff", size=12), name="Rescue base station", showlegend=True, hoverinfo="name"))
    return tr


def _legend_proxies() -> list:
    """Empty traces that only exist to give the legend explanatory entries."""
    spec = [("UAV", "diamond", "#38bdf8"), ("Victim (ground truth)", "circle", "#ef4444"), ("Estimated position (GenNF)", "x", "#fb923c"),
            ("First-aid delivery point", "diamond-open", "#22c55e")]
    return [go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=n, marker=dict(symbol=s, color=c, size=6)) for n, s, c in spec]


def _nan_join(segments: list[np.ndarray]):
    xs, ys, zs = [], [], []
    for seg in segments:
        xs += list(seg[:, 0]) + [None]; ys += list(seg[:, 1]) + [None]; zs += list(seg[:, 2]) + [None]
    return xs, ys, zs


def dynamic_traces(sim: Simulation, k: int, trail_len: int = 40) -> list:
    h = sim.history[k]; pos = h["pos"]; n = len(pos); names = [u.name for u in sim.net.uavs]
    hover = [f"{names[i]}<br>{h['uav_status'][i]}<br>alt {pos[i, 2]:.0f} m<br>battery {h['battery'][i]:.0f} % (simulated)<br>"
             f"comm {'OK' if h['comm'][i] else 'LOST'}" for i in range(n)]
    uav = go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text", text=[nm[-2:] for nm in names],
                       textposition="top center", textfont=dict(color="#e0f2fe", size=10), hovertext=hover, hoverinfo="text", name="UAV",
                       marker=dict(size=6, symbol="diamond", color=[UAV_COLORS[s] for s in h["uav_status"]],
                                   line=dict(color=["#ffffff" if c else "#ef4444" for c in h["comm"]], width=2)), showlegend=False)
    lo = max(0, k - trail_len)
    trails = _nan_join([np.array([sim.history[j]["pos"][i] for j in range(lo, k + 1)]) for i in range(n)])
    trail = go.Scatter3d(x=trails[0], y=trails[1], z=trails[2], mode="lines", line=dict(color="rgba(125,211,252,0.55)", width=2),
                         hoverinfo="skip", showlegend=False, name="Flight paths")
    routes, links, errs = [], [], []
    vt = sim.victims
    for vi, v in enumerate(vt):
        e = h["v_est"][vi]
        if e is not None:
            errs.append(np.array([v.true_position, e]))
            for ui, st in enumerate(h["uav_status"]):
                if st in (UAVStatus.TRANSIT.value, UAVStatus.DELIVERING.value) and sim.net.uavs[ui].victim_id == vi:
                    routes.append(np.array([pos[ui], e]))
        for a in h["v_links"][vi]:
            links.append(np.array([pos[a], v.true_position]))
    r = _nan_join(routes) if routes else ([None], [None], [None])
    route = go.Scatter3d(x=r[0], y=r[1], z=r[2], mode="lines", line=dict(color="#f59e0b", width=4, dash="dash"), hoverinfo="skip", showlegend=False)
    l = _nan_join(links) if links else ([None], [None], [None])
    link = go.Scatter3d(x=l[0], y=l[1], z=l[2], mode="lines", line=dict(color="#22d3ee", width=2, dash="dot"), hoverinfo="skip", showlegend=False)
    er = _nan_join(errs) if errs else ([None], [None], [None])
    err = go.Scatter3d(x=er[0], y=er[1], z=er[2], mode="lines", line=dict(color="#fb923c", width=3), hoverinfo="skip", showlegend=False)
    tp = np.array([v.true_position for v in vt]).reshape(-1, 3)
    vh = [f"{v.name} | urgency {v.urgency} | {v.supply}<br>status: {h['v_status'][i]}" + (f"<br>error {h['v_err'][i]:.1f} m" if h['v_err'][i] is not None else "")
          for i, v in enumerate(vt)]
    truth = go.Scatter3d(x=tp[:, 0], y=tp[:, 1], z=tp[:, 2], mode="markers+text", text=[v.name for v in vt], textposition="bottom center",
                         textfont=dict(color="#fecaca", size=10), hovertext=vh, hoverinfo="text", showlegend=False,
                         marker=dict(size=6, color=[VICTIM_COLORS[s] for s in h["v_status"]], line=dict(color="white", width=1)))
    idx = [i for i, e in enumerate(h["v_est"]) if e is not None]
    ep = np.array([h["v_est"][i] for i in idx]).reshape(-1, 3)
    est = go.Scatter3d(x=ep[:, 0], y=ep[:, 1], z=ep[:, 2], mode="markers", showlegend=False, hoverinfo="text",
                       hovertext=[f"{vt[i].name} estimate (error {h['v_err'][i]:.1f} m)" for i in idx],
                       marker=dict(size=6, symbol="x", color="#fb923c", line=dict(color="#fb923c", width=2)))
    dp = np.array([vt[i].true_position for i in range(len(vt)) if h["v_assisted"][i]]).reshape(-1, 3)
    deliv = go.Scatter3d(x=dp[:, 0], y=dp[:, 1], z=dp[:, 2] + 2, mode="markers", showlegend=False, hoverinfo="skip",
                         marker=dict(size=9, symbol="diamond-open", color="#22c55e", line=dict(width=3)))
    return [uav, trail, route, link, truth, est, err, deliv]


def build_figure(sim: Simulation, max_frames: int = 150, frame_ms: int = 80, height: int = 720) -> go.Figure:
    st = static_traces(sim.env) + _legend_proxies(); S = len(st)
    n_hist = len(sim.history); stride = max(1, int(np.ceil(n_hist / max_frames)))
    ks = list(range(0, n_hist, stride)); ks += [n_hist - 1] if ks[-1] != n_hist - 1 else []
    first = dynamic_traces(sim, ks[0])
    fig = go.Figure(data=st + first)
    dyn_idx = list(range(S, S + len(first)))
    fig.frames = [go.Frame(data=dynamic_traces(sim, k), traces=dyn_idx, name=str(k)) for k in ks]
    s = sim.env.size
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0b1220", font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="rgba(15,23,42,0.7)", x=0.01, y=0.99),
        scene=dict(xaxis=dict(range=[0, s], title="x (m)", backgroundcolor="#0b1220", gridcolor="#1e293b", color="#94a3b8"),
                   yaxis=dict(range=[0, s], title="y (m)", backgroundcolor="#0b1220", gridcolor="#1e293b", color="#94a3b8"),
                   zaxis=dict(range=[-1, 70], title="z (m)", backgroundcolor="#0b1220", gridcolor="#1e293b", color="#94a3b8"),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.3), camera=dict(eye=dict(x=1.35, y=-1.35, z=0.85))),
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.02, bgcolor="#1e293b", font=dict(color="#e2e8f0"), buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.18, len=0.8, y=0.02, currentvalue=dict(prefix="t = ", suffix=" s", font=dict(color="#e2e8f0")),
                      font=dict(color="#94a3b8"), steps=[dict(label=f"{sim.history[k]['t']:.0f}", method="animate",
                      args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for k in ks])])
    return fig


# ===== DARK THEME FOR SCIENTIFIC CHARTS =====
def style_chart(fig, height=210):
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="#22252A",
        plot_bgcolor="#22252A",
        font=dict(family="Inter, sans-serif", size=11, color="#E8EAED"),
        margin=dict(l=38, r=12, t=38, b=36),
        legend=dict(font=dict(size=10, color="#B0B7C3")),
    )
    fig.update_xaxes(gridcolor="#414750", zerolinecolor="#515966", linecolor="#414750",
                     tickfont=dict(size=10, color="#B0B7C3"), title_font=dict(size=11, color="#B0B7C3"))
    fig.update_yaxes(gridcolor="#414750", zerolinecolor="#515966", linecolor="#414750",
                     tickfont=dict(size=10, color="#B0B7C3"), title_font=dict(size=11, color="#B0B7C3"))
    return fig

# ------------------------------------------------------------------ dashboard charts
def _layout(fig: go.Figure, title: str, h: int = 220) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=12)), height=h, margin=dict(l=36, r=10, t=30, b=30),
                      paper_bgcolor="#292D33", plot_bgcolor="#22252A", font=dict(color="#E8EAED", size=10),
                      template="plotly_dark")
    return fig


def chart_true_vs_est(sim: Simulation, k: int) -> go.Figure:
    h = sim.history[k]; fig = go.Figure()
    idx = [i for i, e in enumerate(h["v_est"]) if e is not None]
    t = np.array([sim.victims[i].true_position for i in idx]); e = np.array([h["v_est"][i] for i in idx])
    fig.add_scatter(x=t[:, 0] if len(idx) else [], y=t[:, 1] if len(idx) else [], mode="markers", name="true", marker=dict(size=10, color="#ef4444"))
    fig.add_scatter(x=e[:, 0] if len(idx) else [], y=e[:, 1] if len(idx) else [], mode="markers", name="estimated", marker=dict(size=10, symbol="x", color="#fb923c"))
    for a, b in zip(t, e): fig.add_scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color="#64748b", width=1), showlegend=False)
    fig.update_xaxes(range=[0, sim.env.size], title="x (m)"); fig.update_yaxes(range=[0, sim.env.size], title="y (m)", scaleanchor="x")
    return _layout(fig, "True vs estimated victim positions (top view)", 260)


def chart_errors(sim: Simulation, k: int) -> go.Figure:
    h = sim.history[k]; names = [v.name for v, e in zip(sim.victims, h["v_err"]) if e is not None]; vals = [e for e in h["v_err"] if e is not None]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=["#22c55e" if e <= sim.cfg.success_threshold_m else "#fb923c" for e in vals]))
    fig.add_hline(y=sim.cfg.success_threshold_m, line_dash="dash", annotation_text="success threshold")
    fig.update_yaxes(title="3D error (m)")
    return _layout(fig, "Localization error per victim")


def chart_rssi(sim: Simulation, victim_idx: int) -> go.Figure:
    v = sim.victims[victim_idx]; fig = go.Figure()
    if v.last_rssi is not None:
        n = len(v.last_rssi); sel = set(v.anchor_ids)
        fig.add_bar(x=[f"UAV-{i + 1:02d}" for i in range(n)], y=np.nan_to_num(v.last_rssi, nan=np.nan),
                    marker_color=["#22d3ee" if i in sel else "#475569" for i in range(n)])
    fig.update_yaxes(title="RSSI (dBm)")
    return _layout(fig, f"RSSI at each UAV for {v.name} (cyan = selected anchors; missing = out of range)")


def chart_uav_status(sim: Simulation, k: int) -> go.Figure:
    s = sim.history[k]["uav_status"]; keys = list(UAV_COLORS)
    fig = go.Figure(go.Bar(x=keys, y=[s.count(c) for c in keys], marker_color=[UAV_COLORS[c] for c in keys]))
    fig.update_yaxes(title="UAVs", dtick=1)
    return _layout(fig, "UAV mission status")


def chart_missions(sim: Simulation, k: int) -> go.Figure:
    s = sim.history[k]["v_status"]
    done = s.count("assisted"); failed = s.count("failed"); pend = len(s) - done - failed
    fig = go.Figure(go.Pie(labels=["Completed", "Pending", "Failed"], values=[done, pend, failed], hole=0.55,
                           marker=dict(colors=["#22c55e", "#f59e0b", "#111827"])))
    return _layout(fig, "Completed vs pending rescue missions")


def chart_noise_sweep(df) -> go.Figure:
    g = df.groupby(["noise_db", "method"]).error_m.mean().reset_index(); fig = go.Figure()
    for m, gg in g.groupby("method"): fig.add_scatter(x=gg.noise_db, y=gg.error_m, mode="lines+markers", name=m)
    fig.update_xaxes(title="RSSI noise σ (dB)"); fig.update_yaxes(title="mean 3D error (m)")
    return _layout(fig, "Localization error vs RSSI noise", 380)
