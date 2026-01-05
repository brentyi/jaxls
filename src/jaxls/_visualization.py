"""Visualization utilities for jaxls.

This module provides tools for visualizing problem structure using
an interactive D3.js force-directed graph.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._problem import LeastSquaresProblem


def _get_cost_color(kind: str) -> tuple[str, str]:
    """Get fill color and border color for a cost node based on its kind.

    Returns:
        Tuple of (fill_color, border_color) for graphviz.
    """
    colors = {
        "l2_squared": ("#a8d5ff", "#4a90d9"),
        "constraint_eq_zero": ("#a8ffa8", "#4ad94a"),
        "constraint_leq_zero": ("#ffe4a8", "#d9a84a"),
        "constraint_geq_zero": ("#ffa8e4", "#d94aa8"),
    }
    return colors.get(kind, ("#e0e0e0", "#808080"))


VAR_COLORS = [
    ("#ffcccc", "#cc6666"),  # Red
    ("#ccccff", "#6666cc"),  # Blue
    ("#ccffcc", "#66cc66"),  # Green
    ("#ffd9b3", "#cc8033"),  # Orange
    ("#ffccff", "#cc66cc"),  # Magenta
    ("#ccffff", "#66cccc"),  # Cyan
    ("#ffffcc", "#cccc66"),  # Yellow
    ("#d9b3ff", "#8033cc"),  # Purple
]


def _get_var_color(var_type_name: str, var_type_index: int) -> tuple[str, str]:
    """Get fill color and border color for a variable node based on type index.

    Args:
        var_type_name: Name of the variable type (unused, kept for compatibility).
        var_type_index: Index of this type in the sorted list of all variable types.

    Returns:
        Tuple of (fill_color, border_color).
    """
    return VAR_COLORS[var_type_index % len(VAR_COLORS)]


def _problem_to_graph_data(
    problem: LeastSquaresProblem,
    max_costs: int | None = None,
    max_variables: int | None = None,
) -> dict[str, Any]:
    """Convert problem to graph data format for D3.js visualization.

    Args:
        problem: The least squares problem to visualize.
        max_costs: Maximum number of cost nodes to show. If None, show all.
            When multiple cost types exist, the limit is distributed proportionally.
        max_variables: Maximum number of variables per type to show. If None, show all.

    Returns:
        Dict with 'nodes', 'links' lists, and 'truncated' flag.
    """
    import numpy as onp

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    costs = list(problem.costs)

    # Pre-pass: collect all unique variable type names for consistent color assignment.
    all_var_types: set[str] = set()
    for cost in costs:
        for var in cost._get_variables():
            all_var_types.add(type(var).__name__)
    sorted_var_types = sorted(all_var_types)
    var_type_to_index = {name: i for i, name in enumerate(sorted_var_types)}

    # First pass: collect all unique variables, respecting max_variables per type.
    var_nodes: dict[tuple[str, int], int] = {}  # (type_name, id) -> node_index
    vars_count_by_type: dict[str, int] = {}  # Track count per type.
    truncated_vars = False

    for cost in costs:
        for var in cost._get_variables():
            var_type_name = type(var).__name__
            ids = onp.atleast_1d(onp.array(var.id)).flatten()
            for var_id in ids:
                key = (var_type_name, int(var_id))
                if key not in var_nodes:
                    # Check if we've hit the limit for this type.
                    current_count = vars_count_by_type.get(var_type_name, 0)
                    if max_variables is not None and current_count >= max_variables:
                        truncated_vars = True
                        continue

                    type_idx = var_type_to_index[var_type_name]
                    fill_color, border_color = _get_var_color(var_type_name, type_idx)
                    node_idx = len(nodes)
                    var_nodes[key] = node_idx
                    vars_count_by_type[var_type_name] = current_count + 1
                    nodes.append({
                        "id": f"var_{node_idx}",
                        "label": f"{var_type_name}({var_id})",
                        "type": "variable",
                        "var_type": var_type_name,
                        "fill": fill_color,
                        "stroke": border_color,
                    })

    # Count total cost nodes per cost type (by name) for proportional allocation.
    cost_counts_by_name: dict[str, int] = {}
    for cost in costs:
        name = cost._get_name()
        variables = cost._get_variables()
        if not variables:
            batch_size = 1
        else:
            batch_sizes = []
            for var in variables:
                ids = onp.atleast_1d(onp.array(var.id)).flatten()
                batch_sizes.append(len(ids))
            batch_size = max(batch_sizes)
        cost_counts_by_name[name] = cost_counts_by_name.get(name, 0) + batch_size

    # Calculate per-type limits for proportional distribution.
    if max_costs is not None and cost_counts_by_name:
        num_cost_types = len(cost_counts_by_name)
        per_type_limit = max_costs // num_cost_types
        # Distribute remainder to types with more costs.
        remainder = max_costs % num_cost_types
        sorted_names = sorted(
            cost_counts_by_name.keys(),
            key=lambda n: cost_counts_by_name[n],
            reverse=True,
        )
        max_per_cost_type: dict[str, int] = {}
        for i, name in enumerate(sorted_names):
            max_per_cost_type[name] = per_type_limit + (1 if i < remainder else 0)
    else:
        max_per_cost_type = None

    # Second pass: add cost nodes and edges.
    # Only include costs where ALL variables are visible.
    cost_node_idx = 0
    costs_added_by_name: dict[str, int] = {}
    truncated_costs = False

    for cost in costs:
        name = cost._get_name()
        fill_color, border_color = _get_cost_color(cost.kind)
        variables = cost._get_variables()

        var_id_arrays = []
        for var in variables:
            ids = onp.atleast_1d(onp.array(var.id)).flatten()
            var_id_arrays.append((type(var).__name__, ids))

        batch_sizes = [len(ids) for _, ids in var_id_arrays]
        max_batch = max(batch_sizes) if len(batch_sizes) > 0 else 1

        expanded_ids = []
        for var_type_name, ids in var_id_arrays:
            if len(ids) == 1 and max_batch > 1:
                expanded_ids.append((var_type_name, onp.repeat(ids, max_batch)))
            else:
                expanded_ids.append((var_type_name, ids))

        for batch_idx in range(max_batch):
            # Check if we've hit the per-type limit.
            current_type_count = costs_added_by_name.get(name, 0)
            if max_per_cost_type is not None:
                if current_type_count >= max_per_cost_type[name]:
                    truncated_costs = True
                    continue

            # Check if all variables for this cost are visible.
            all_vars_visible = True
            var_keys = []
            for var_type_name, ids in expanded_ids:
                var_id = int(ids[batch_idx])
                key = (var_type_name, var_id)
                if key not in var_nodes:
                    all_vars_visible = False
                    break
                var_keys.append(key)

            if not all_vars_visible:
                continue

            node_id = f"cost_{cost_node_idx}"
            cost_node_idx += 1
            costs_added_by_name[name] = current_type_count + 1

            nodes.append({
                "id": node_id,
                "label": name,
                "type": "cost",
                "kind": cost.kind,
                "fill": fill_color,
                "stroke": border_color,
            })

            for key in var_keys:
                target_idx = var_nodes[key]
                links.append({
                    "source": node_id,
                    "target": nodes[target_idx]["id"],
                })

    return {
        "nodes": nodes,
        "links": links,
        "truncated": truncated_vars or truncated_costs,
    }


def _count_total_costs(problem: LeastSquaresProblem) -> int:
    """Count total number of cost nodes (including batched costs)."""
    import numpy as onp

    total = 0
    for cost in problem.costs:
        variables = cost._get_variables()
        if not variables:
            total += 1
            continue
        batch_sizes = []
        for var in variables:
            ids = onp.atleast_1d(onp.array(var.id)).flatten()
            batch_sizes.append(len(ids))
        total += max(batch_sizes)
    return total


def _count_total_variables(problem: LeastSquaresProblem) -> dict[str, int]:
    """Count total number of unique variables by type."""
    import numpy as onp

    var_ids_by_type: dict[str, set[int]] = {}
    for cost in problem.costs:
        for var in cost._get_variables():
            var_type_name = type(var).__name__
            ids = onp.atleast_1d(onp.array(var.id)).flatten()
            if var_type_name not in var_ids_by_type:
                var_ids_by_type[var_type_name] = set()
            var_ids_by_type[var_type_name].update(int(i) for i in ids)
    return {k: len(v) for k, v in var_ids_by_type.items()}


def problem_show(
    problem: LeastSquaresProblem,
    *,
    width: int = 800,
    height: int = 500,
    max_costs: int = 1000,
    max_variables: int = 500,
) -> None:
    """Display an interactive graph showing costs and variables.

    Uses Canvas rendering for better performance with large graphs.

    Args:
        problem: The least squares problem to visualize.
        width: Width of the visualization in pixels.
        height: Height of the visualization in pixels.
        max_costs: Maximum number of cost nodes to show. When multiple cost
            types exist, the limit is distributed proportionally across types.
        max_variables: Maximum number of variables per type to show.
            Only costs where all variables are visible are shown.
    """
    # Count totals before filtering.
    total_costs = _count_total_costs(problem)
    total_vars_by_type = _count_total_variables(problem)
    total_vars = sum(total_vars_by_type.values())

    graph_data = _problem_to_graph_data(problem, max_costs, max_variables)
    graph_json = json.dumps(graph_data)

    # Count displayed nodes.
    displayed_costs = sum(1 for n in graph_data["nodes"] if n["type"] == "cost")
    displayed_vars = sum(1 for n in graph_data["nodes"] if n["type"] == "variable")

    # Build status message.
    if graph_data["truncated"]:
        status_msg = f"Showing {displayed_costs}/{total_costs} costs, {displayed_vars}/{total_vars} variables"
    else:
        status_msg = f"{displayed_costs} costs, {displayed_vars} variables"

    # Build inner HTML document for iframe.
    inner_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #container {{ width: 100%; height: 100vh; position: relative; }}
        canvas {{ display: block; width: 100%; height: 100%; }}
        .status {{ position: absolute; bottom: 5px; right: 10px; font-size: 10px; color: #666; font-family: Helvetica, Arial, sans-serif; background: rgba(255,255,255,0.8); padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="container">
        <canvas></canvas>
        <div class="status">{status_msg}</div>
    </div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {{
        const data = {graph_json};
        const container = d3.select("#container");
        const canvas = container.select("canvas").node();
        const ctx = canvas.getContext("2d");

    // Get actual container dimensions.
    const rect = container.node().getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Node sizing.
    const varRx = 45, varRy = 14;  // Ellipse radii for variables.
    const costW = 90, costH = 22;   // Rect size for costs.

    // Set up high-DPI canvas.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Track transform for zoom/pan.
    let transform = d3.zoomIdentity;

    // Create force simulation.
    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(60))
        .force("charge", d3.forceManyBody().strength(-150).distanceMax(250))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(35));

    // Drag state.
    let dragNode = null;

    // Find node at position.
    function nodeAt(x, y) {{
        const tx = (x - transform.x) / transform.k;
        const ty = (y - transform.y) / transform.k;
        for (let i = data.nodes.length - 1; i >= 0; i--) {{
            const d = data.nodes[i];
            const dx = tx - d.x;
            const dy = ty - d.y;
            if (d.type === "variable") {{
                if ((dx*dx)/(varRx*varRx) + (dy*dy)/(varRy*varRy) < 1) return d;
            }} else {{
                if (Math.abs(dx) < costW/2 && Math.abs(dy) < costH/2) return d;
            }}
        }}
        return null;
    }}

    function draw() {{
        ctx.save();
        ctx.clearRect(0, 0, width, height);
        ctx.translate(transform.x, transform.y);
        ctx.scale(transform.k, transform.k);

        // Draw links.
        ctx.strokeStyle = "#999";
        ctx.globalAlpha = 0.6;
        ctx.lineWidth = 1.5 / transform.k;
        ctx.beginPath();
        data.links.forEach(d => {{
            ctx.moveTo(d.source.x, d.source.y);
            ctx.lineTo(d.target.x, d.target.y);
        }});
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Draw nodes.
        data.nodes.forEach(d => {{
            ctx.save();
            ctx.translate(d.x, d.y);

            if (d.type === "variable") {{
                // Ellipse for variables.
                ctx.beginPath();
                ctx.ellipse(0, 0, varRx, varRy, 0, 0, 2 * Math.PI);
                ctx.fillStyle = d.fill;
                ctx.fill();
                ctx.strokeStyle = d.stroke;
                ctx.lineWidth = 1.5 / transform.k;
                ctx.stroke();
            }} else {{
                // Rounded rect for costs.
                const w = costW, h = costH, r = 4;
                ctx.beginPath();
                ctx.moveTo(-w/2 + r, -h/2);
                ctx.lineTo(w/2 - r, -h/2);
                ctx.quadraticCurveTo(w/2, -h/2, w/2, -h/2 + r);
                ctx.lineTo(w/2, h/2 - r);
                ctx.quadraticCurveTo(w/2, h/2, w/2 - r, h/2);
                ctx.lineTo(-w/2 + r, h/2);
                ctx.quadraticCurveTo(-w/2, h/2, -w/2, h/2 - r);
                ctx.lineTo(-w/2, -h/2 + r);
                ctx.quadraticCurveTo(-w/2, -h/2, -w/2 + r, -h/2);
                ctx.closePath();
                ctx.fillStyle = d.fill;
                ctx.fill();
                ctx.strokeStyle = d.stroke;
                ctx.lineWidth = 1.5 / transform.k;
                ctx.stroke();
            }}

            // Draw label.
            ctx.fillStyle = "#000";
            ctx.font = "9px Helvetica, Arial, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(d.label, 0, 0);

            ctx.restore();
        }});

        // Draw legend.
        ctx.restore();
        drawLegend();
    }}

    function drawLegend() {{
        // Build legend dynamically from actual nodes.
        const legendData = [];

        // Collect unique cost kinds.
        const costKindLabels = {{
            "l2_squared": "Cost (L2)",
            "constraint_eq_zero": "Equality",
            "constraint_leq_zero": "Inequality (≤)",
            "constraint_geq_zero": "Inequality (≥)"
        }};
        const seenCostKinds = new Set();
        data.nodes.filter(n => n.type === "cost").forEach(n => seenCostKinds.add(n.kind));
        for (const kind of ["l2_squared", "constraint_eq_zero", "constraint_leq_zero", "constraint_geq_zero"]) {{
            if (seenCostKinds.has(kind)) {{
                const node = data.nodes.find(n => n.type === "cost" && n.kind === kind);
                legendData.push({{
                    label: costKindLabels[kind],
                    fill: node.fill,
                    stroke: node.stroke,
                    type: "cost"
                }});
            }}
        }}

        // Collect unique variable types.
        const seenVarTypes = new Map();  // var_type -> {{fill, stroke}}
        data.nodes.filter(n => n.type === "variable").forEach(n => {{
            if (!seenVarTypes.has(n.var_type)) {{
                seenVarTypes.set(n.var_type, {{ fill: n.fill, stroke: n.stroke }});
            }}
        }});
        for (const [varType, colors] of seenVarTypes) {{
            legendData.push({{
                label: varType,
                fill: colors.fill,
                stroke: colors.stroke,
                type: "variable"
            }});
        }}

        ctx.save();
        ctx.translate(10, 10);

        legendData.forEach((d, i) => {{
            const y = i * 18;
            if (d.type === "variable") {{
                ctx.beginPath();
                ctx.ellipse(12, y + 6, 12, 6, 0, 0, 2 * Math.PI);
                ctx.fillStyle = d.fill;
                ctx.fill();
                ctx.strokeStyle = d.stroke;
                ctx.lineWidth = 1;
                ctx.stroke();
            }} else {{
                ctx.beginPath();
                ctx.roundRect(0, y, 24, 12, 2);
                ctx.fillStyle = d.fill;
                ctx.fill();
                ctx.strokeStyle = d.stroke;
                ctx.lineWidth = 1;
                ctx.stroke();
            }}
            ctx.fillStyle = "#000";
            ctx.font = "9px Helvetica, Arial, sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(d.label, 30, y + 6);
        }});

        ctx.restore();
    }}

    simulation.on("tick", draw);

    // Zoom behavior with filter to allow node dragging.
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .filter((event) => {{
            // Block zoom if clicking on a node (allow node drag instead).
            if (event.type === "mousedown") {{
                const [x, y] = d3.pointer(event);
                if (nodeAt(x, y)) return false;
            }}
            return true;
        }})
        .on("zoom", (event) => {{
            transform = event.transform;
            draw();
        }});

    d3.select(canvas)
        .call(zoom)
        .on("mousedown.drag", (event) => {{
            const [x, y] = d3.pointer(event);
            const node = nodeAt(x, y);
            if (node) {{
                dragNode = node;
                simulation.alphaTarget(0.3).restart();
                dragNode.fx = dragNode.x;
                dragNode.fy = dragNode.y;
            }}
        }})
        .on("mousemove.drag", (event) => {{
            if (dragNode) {{
                const [x, y] = d3.pointer(event);
                dragNode.fx = transform.invertX(x);
                dragNode.fy = transform.invertY(y);
                draw();
            }}
        }})
        .on("mouseup.drag mouseleave.drag", () => {{
            if (dragNode) {{
                simulation.alphaTarget(0);
                dragNode.fx = null;
                dragNode.fy = null;
                dragNode = null;
            }}
        }});

        // Initial draw.
        draw();
    }})();
    </script>
</body>
</html>"""

    # Try to display in Jupyter notebook/lab if available.
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        # Check for Jupyter kernel (ZMQInteractiveShell), not terminal IPython.
        if ipython is not None and "ZMQInteractiveShell" in type(ipython).__name__:
            data_uri = "data:text/html;base64," + base64.b64encode(
                inner_html.encode("utf-8")
            ).decode("ascii")

            from IPython.display import IFrame, display

            display(IFrame(src=data_uri, width="100%", height=height))
            return
    except ImportError:
        pass

    # Fallback: open in default web browser.
    import tempfile
    import webbrowser

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(inner_html)
        webbrowser.open("file://" + f.name)
