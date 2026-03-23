"""
modules/pmi.py
==============
Interactive PMI tag co-occurrence graph, rendered as a Streamlit v2 component.

Design
------
Uses st.components.v2.component() with inline HTML/CSS/JS — no files written
to disk, no file watcher, no iframe boundary. The JS runs directly in the
Streamlit page DOM and communicates back to Python via the v2 state API:

    setStateValue('tags', anchorTags)   →  on_tags_change() callback fires

The component is registered once at module import time (_pmi_graph_component).
PMI data (all_tags, full_pmi, pmi_max, init_anchors) is passed as data= at
mount time; the JS function runs once on mount and owns anchor state from then on.

Two-way sync
------------
- Python → Graph:  init_anchors seeds anchorTags on mount (stable key ensures
                   the JS is not re-executed on reruns).
- Graph → Python:  every addAnchor() call fires setStateValue(), which triggers
                   a Streamlit rerun. The on_tags_change callback reads the new
                   tag list from session_state and writes it to selected_tags
                   before any widgets render.

Graph modes
-----------
- No anchors: renders the top-150 tags by PMI degree (full-graph mode, green nodes).
- With anchors: renders the ego-network — anchors (red) + PMI neighbours (blue).
                The left panel lists neighbours ranked by PMI boost strength.

Public API
----------
    render_pmi_explorer(pmi_graph, init_anchors, *, key, on_tags_change)
"""

from __future__ import annotations

import json
import math
from typing import Dict, List

import streamlit as st
import streamlit.components.v2 as components_v2

# ---------------------------------------------------------------------------
# Static HTML / CSS — registered once at module import
# ---------------------------------------------------------------------------

_HTML = """
<div id="pmi-root" style="display:grid; grid-template-columns:200px 1fr; height:680px; overflow:hidden;">
  <aside id="pmi-aside">
    <div class="sidebar-divider" id="neighbor-header" style="display:none;">
      Neighbors · PMI boost
    </div>
    <div class="neighbor-scroll" id="neighbor-list"></div>
  </aside>
  <div id="graph-wrap">
    <svg id="graph-svg" style="width:100%;height:100%;"></svg>
    <div id="empty-state">
      <div class="big">NO TAGS</div>
      <div class="hint">Add tags in the control panel to explore</div>
    </div>
    <div id="pmi-tooltip"></div>
    <div id="pmi-stats"></div>
  </div>
</div>
"""

_CSS = """
:root {
  --bg:         #0d0d14;
  --surface:    #13131f;
  --border:     #1e1e30;
  --anchor:     #ff4060;
  --anchor-dim: #7a1f2e;
  --neighbor:   #3b82f6;
  --text:       #e2e2f0;
  --muted:      #555570;
  --accent:     #ff4060;
  --tag-pill:   #1a1a2e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
#pmi-root {
  background: var(--bg);
  color: var(--text);
  font-family: 'Syne', 'Segoe UI', sans-serif;
  border-radius: 6px;
  overflow: hidden;
}
#pmi-aside {
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
}
.sidebar-divider {
  font-family: 'Space Mono', monospace; font-size: 0.58rem;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em;
  padding: 8px 12px 3px; border-top: 1px solid var(--border);
}
.neighbor-scroll { flex:1; overflow-y:auto; padding: 2px 0; }
.neighbor-scroll::-webkit-scrollbar { width: 3px; }
.neighbor-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.neighbor-row {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 12px; cursor: pointer;
  border-left: 2px solid transparent; transition: background 0.1s;
}
.neighbor-row:hover { background: var(--tag-pill); border-left-color: var(--neighbor); }
.neighbor-label { font-size: 0.72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 110px; }
.neighbor-bar-wrap { flex:1; height:3px; background:var(--border); border-radius:2px; overflow:hidden; }
.neighbor-bar { height:100%; background:var(--neighbor); border-radius:2px; transition: width 0.3s ease; }
.neighbor-pmi { font-family:'Space Mono',monospace; font-size:0.58rem; color:var(--muted); white-space:nowrap; }
#graph-wrap { position:relative; overflow:hidden; background:var(--bg); }
#empty-state {
  position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:10px; pointer-events:none;
}
#empty-state .big { font-size:2rem; opacity:0.07; font-weight:800; letter-spacing:0.05em; }
#empty-state .hint { font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted); }
#pmi-tooltip {
  position:absolute; background:var(--surface); border:1px solid var(--border);
  border-radius:5px; padding:9px 12px; font-family:'Space Mono',monospace;
  font-size:0.65rem; color:var(--text); pointer-events:none;
  opacity:0; transition:opacity 0.15s; z-index:200; max-width:200px; line-height:1.7;
}
#pmi-tooltip.visible { opacity:1; }
#pmi-tooltip strong { color:var(--accent); display:block; margin-bottom:3px; font-size:0.78rem; }
#pmi-stats {
  position:absolute; bottom:12px; right:12px;
  font-family:'Space Mono',monospace; font-size:0.58rem;
  color:var(--muted); text-align:right; line-height:1.8;
}
"""

_JS = r"""
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

export default function(component) {
  const { data, setStateValue, parentElement } = component;

  // data = { all_tags, full_pmi, pmi_max, init_anchors }
  const ALL_TAGS   = data.all_tags;
  const FULL_PMI   = data.full_pmi;
  const PMI_MAX    = data.pmi_max;

  const NODE_R_ANCHOR  = 20, NODE_R_MIN = 5, NODE_R_MAX = 16;
  const COLOR_ANCHOR   = '#ff4060';
  const COLOR_NEIGHBOR = '#3b82f6';
  const COLOR_FULL     = '#59a14f';
  const COLOR_EDGE_AN  = '#ff406040';
  const COLOR_EDGE_AA  = '#ff406080';
  const FULL_GRAPH_TOP_N = 150;

  let anchorTags = [...(data.init_anchors || [])];
  let simulation = null;

  // ── Anchor management ───────────────────────────────────────────────
  // anchorTags is always driven by the control panel multiselect (init_anchors).
  // Clicking a node or neighbor row sends the new list to Python via setStateValue,
  // which updates the multiselect — the single source of truth.
  function addAnchor(tag) {
    if (anchorTags.includes(tag) || !ALL_TAGS.includes(tag)) return;
    anchorTags.push(tag);
    refreshAll();
    setStateValue('tags', anchorTags.slice());
  }

  // ── Graph data ──────────────────────────────────────────────────────
  function buildFullGraphData() {
    const byDegree = Object.entries(FULL_PMI)
      .map(([tag, nbrs]) => [tag, Object.keys(nbrs).length])
      .sort((a, b) => b[1] - a[1]).slice(0, FULL_GRAPH_TOP_N);
    const keepSet = new Set(byDegree.map(([t]) => t));
    const maxDeg  = byDegree.length ? byDegree[0][1] : 1;
    const nodes = {}, edges = [], seen = new Set();
    for (const [tag, deg] of byDegree)
      nodes[tag] = { id: tag, label: tag, role: 'full', size: deg / maxDeg, pmi: null };
    for (const tag of keepSet)
      for (const [nbr, pmi] of Object.entries(FULL_PMI[tag] || {})) {
        if (!keepSet.has(nbr)) continue;
        const key = [tag, nbr].sort().join('|||');
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push({ source: tag, target: nbr, pmi, weight: pmi / PMI_MAX });
      }
    return { nodes: Object.values(nodes), edges, anchor_count: 0, neighbor_count: 0, full_mode: true };
  }

  function buildGraphData(anchors) {
    const valid = anchors.filter(t => t in FULL_PMI);
    if (!valid.length) return buildFullGraphData();
    const nodes = {}, edges = [], seen = new Set(), neighborMax = {};
    for (const a of valid)
      nodes[a] = { id: a, label: a, role: 'anchor', size: 1.0, pmi: null };
    for (const anchor of valid)
      for (const [nbr, rawPmi] of Object.entries(FULL_PMI[anchor] || {})) {
        if (!(nbr in nodes)) neighborMax[nbr] = Math.max(neighborMax[nbr] || 0, rawPmi / PMI_MAX);
        const key = [anchor, nbr].sort().join('|||');
        if (!seen.has(key)) { seen.add(key); edges.push({ source: anchor, target: nbr, pmi: rawPmi, weight: rawPmi / PMI_MAX }); }
      }
    for (let i = 0; i < valid.length; i++)
      for (let j = i + 1; j < valid.length; j++) {
        const p = (FULL_PMI[valid[i]] || {})[valid[j]];
        if (p != null) { const k = [valid[i], valid[j]].sort().join('|||'); if (!seen.has(k)) { seen.add(k); edges.push({ source: valid[i], target: valid[j], pmi: p, weight: p / PMI_MAX }); } }
      }
    for (const [nbr, normPmi] of Object.entries(neighborMax))
      if (!(nbr in nodes)) nodes[nbr] = { id: nbr, label: nbr, role: 'neighbor', size: normPmi, pmi: normPmi * PMI_MAX };
    return { nodes: Object.values(nodes), edges, anchor_count: valid.length, neighbor_count: Object.keys(neighborMax).length };
  }

  // ── Neighbor list ───────────────────────────────────────────────────
  function renderNeighborList(nodes) {
    const header = parentElement.querySelector('#neighbor-header');
    const list   = parentElement.querySelector('#neighbor-list');
    if (!header || !list) return;
    const neighbors = nodes.filter(n => n.role === 'neighbor').sort((a, b) => b.size - a.size);
    if (!neighbors.length) { header.style.display = 'none'; list.innerHTML = ''; return; }
    header.style.display = 'block';
    list.innerHTML = neighbors.map(n => `
      <div class="neighbor-row" data-tag="${n.id.replace(/"/g, '&quot;')}">
        <span class="neighbor-label" title="${n.id}">${n.id}</span>
        <div class="neighbor-bar-wrap"><div class="neighbor-bar" style="width:${Math.round(n.size*100)}%"></div></div>
        <span class="neighbor-pmi">${n.pmi != null ? n.pmi.toFixed(2) : ''}</span>
      </div>`).join('');
    list.querySelectorAll('.neighbor-row[data-tag]').forEach(row => {
      row.addEventListener('click', () => addAnchor(row.getAttribute('data-tag')));
    });
  }

  function updateStats(data) {
    const el = parentElement.querySelector('#pmi-stats');
    if (!el) return;
    if (!data.nodes.length) { el.textContent = ''; return; }
    el.textContent = data.full_mode
      ? `${data.nodes.length} tags · ${data.edges.length} edges · click a node to anchor it`
      : `${data.anchor_count} anchors · ${data.neighbor_count} neighbors · ${data.edges.length} edges`;
  }

  // ── D3 graph ────────────────────────────────────────────────────────
  function renderGraph(data) {
    const svgEl = parentElement.querySelector('#graph-svg');
    const wrap  = parentElement.querySelector('#graph-wrap');
    const empty = parentElement.querySelector('#empty-state');
    if (!svgEl || !wrap) return;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    empty.style.display = data.nodes.length ? 'none' : 'flex';
    if (!data.nodes.length) return;

    const defs = svg.append('defs');
    ['anchor','neighbor','full'].forEach(role => {
      const f = defs.append('filter').attr('id', `pglow-${role}`)
        .attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
      f.append('feGaussianBlur').attr('stdDeviation', role==='anchor'?6:3).attr('result','blur');
      const m = f.append('feMerge');
      m.append('feMergeNode').attr('in','blur');
      m.append('feMergeNode').attr('in','SourceGraphic');
    });

    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.2,4]).on('zoom', e => g.attr('transform', e.transform)));

    const nodes   = data.nodes.map(n => ({...n}));
    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
    const links   = data.edges
      .filter(e => nodeMap[e.source] && nodeMap[e.target])
      .map(e => ({...e, source: nodeMap[e.source], target: nodeMap[e.target]}));

    const nodeRadius = n => n.role==='anchor' ? NODE_R_ANCHOR : NODE_R_MIN + n.size*(NODE_R_MAX-NODE_R_MIN);
    const nodeColor  = n => n.role==='anchor' ? COLOR_ANCHOR : n.role==='neighbor' ? COLOR_NEIGHBOR : COLOR_FULL;

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(n=>n.id)
        .distance(l => nodeRadius(l.source)+nodeRadius(l.target)+40+(1-l.weight)*50).strength(0.6))
      .force('charge', d3.forceManyBody().strength(n => nodeRadius(n)*-8))
      .force('center', d3.forceCenter(W/2, H/2))
      .force('collision', d3.forceCollide().radius(n => nodeRadius(n)+8));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', l => data.full_mode ? '#2a2a44' : l.source.role==='anchor'&&l.target.role==='anchor' ? COLOR_EDGE_AA : COLOR_EDGE_AN)
      .attr('stroke-width', l => 0.5+l.weight*2.5).attr('stroke-linecap','round');

    const node = g.append('g').selectAll('g').data(nodes).join('g').attr('cursor','pointer')
      .call(d3.drag()
        .on('start',(e,d)=>{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
        .on('drag', (e,d)=>{ d.fx=e.x; d.fy=e.y; })
        .on('end',  (e,d)=>{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }));

    node.append('circle').attr('r', n=>nodeRadius(n)*1.4).attr('fill',nodeColor)
      .attr('opacity', n=>n.role==='anchor'?0.12:0.06)
      .attr('filter', n=>`url(#pglow-${n.role==='anchor'?'anchor':'neighbor'})`);
    node.append('circle').attr('r', nodeRadius).attr('fill',nodeColor)
      .attr('fill-opacity', n=>n.role==='anchor'?0.9:0.7)
      .attr('stroke', n=>n.role==='anchor'?'#ff8099':n.role==='neighbor'?'#7eb3ff':'#86c98a')
      .attr('stroke-width',1.5);
    node.append('text').attr('text-anchor','middle').attr('dy', n=>nodeRadius(n)+12)
      .attr('fill', n=>n.role==='anchor'?COLOR_ANCHOR:n.role==='neighbor'?'#a0b8e8':'#86c98a')
      .attr('font-weight', n=>n.role==='anchor'?700:400)
      .attr('font-size', n=>n.role==='anchor'?11:9)
      .attr('pointer-events','none').attr('user-select','none')
      .text(n=>n.label);

    const tooltip = parentElement.querySelector('#pmi-tooltip');
    node
      .on('mouseenter',(e,d)=>{
        const degInfo   = d.role==='full'?`degree: ${Object.keys(FULL_PMI[d.id]||{}).length}<br>`:'';
        const boostInfo = d.role==='neighbor'?`boost: ${(d.size*100).toFixed(1)}%<br>`:'';
        tooltip.innerHTML = `<strong>${d.label}</strong>role: ${d.role}<br>${degInfo}${boostInfo}PMI: ${d.pmi!=null?d.pmi.toFixed(3):'—'}`;
        tooltip.classList.add('visible');
      })
      .on('mousemove',e=>{ tooltip.style.left=(e.offsetX+14)+'px'; tooltip.style.top=(e.offsetY-8)+'px'; })
      .on('mouseleave',()=>tooltip.classList.remove('visible'))
      .on('click',(e,d)=>{ if(d.role==='neighbor'||d.role==='full') addAnchor(d.id); });

    simulation.on('tick',()=>{
      link.attr('x1',l=>l.source.x).attr('y1',l=>l.source.y)
          .attr('x2',l=>l.target.x).attr('y2',l=>l.target.y);
      node.attr('transform',n=>`translate(${n.x},${n.y})`);
    });
  }

  function refreshAll() {
    const data = buildGraphData(anchorTags);
    renderNeighborList(data.nodes);
    renderGraph(data);
    updateStats(data);
  }

  refreshAll();
}
"""

# ---------------------------------------------------------------------------
# Register once at import time — no files, no watcher
# ---------------------------------------------------------------------------

_pmi_graph_component = components_v2.component(
    "pmi_tag_graph",
    html=_HTML,
    css=_CSS,
    js=_JS,
    isolate_styles=False,   # run in page DOM so d3 CDN script executes
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_pmi_explorer(
    pmi_graph: Dict[str, Dict[str, float]],
    init_anchors: List[str] | None = None,
    *,
    key: str = "pmi_explorer",
    on_tags_change=None,
):
    """
    Render the PMI explorer.
    on_tags_change: callable() — fired when the JS calls setStateValue('tags', ...).
    """
    if init_anchors is None:
        init_anchors = []

    all_tags = sorted(pmi_graph.keys())
    pmi_max  = max((w for nb in pmi_graph.values() for w in nb.values()), default=1.0)
    if math.isnan(pmi_max) or pmi_max == 0.0:
        pmi_max = 1.0
    safe_anchors = [t for t in init_anchors if t in pmi_graph]

    return _pmi_graph_component(
        key=key,
        data={
            "all_tags":    all_tags,
            "full_pmi":    dict(pmi_graph),
            "pmi_max":     pmi_max,
            "init_anchors": safe_anchors,
        },
        default={"tags": safe_anchors},
        height=700,
        on_tags_change=on_tags_change if on_tags_change is not None else lambda: None,
    )