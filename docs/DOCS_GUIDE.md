# Documentation Guide

This guide explains how the jaxls documentation is built and maintained.

## Overview

The documentation uses:
- **Sphinx** for building HTML docs
- **myst-nb** for executing and rendering Jupyter notebooks
- **Furo** theme for styling
- **Plotly** for interactive visualizations

## Directory Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main index page
│   ├── _static/css/         # Custom CSS
│   ├── api/                  # API reference RST files
│   └── notebooks/            # Jupyter notebook examples
├── build/                    # Generated output (git-ignored)
└── Makefile                  # Build commands
```

## Building the Documentation

### Prerequisites

Install documentation dependencies:
```bash
pip install sphinx myst-nb furo sphinx-copybutton
```

### Build Commands

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# Clean and rebuild
make clean && make html

# View locally
python -m http.server 8000 --directory build/html
```

## Jupyter Notebooks

### Execution

Notebooks are **automatically executed** during the Sphinx build via myst-nb.
The configuration in `conf.py` sets:
- `nb_execution_mode = "force"` - Always re-execute notebooks
- `nb_execution_timeout = 120` - 2 minute timeout per cell

### Manual Execution

To execute notebooks manually (useful for testing before build):
```bash
cd docs/source/notebooks

# Execute a single notebook
jupyter execute notebook_name.ipynb

# Execute multiple notebooks
jupyter execute *.ipynb
```

### Notebook Style Guidelines

1. **First cell**: Markdown with title, description, and "Features used:" list
2. **Second cell**: Logger setup (hidden with `hide-input` tag)
3. **Code cells**: Use `@jax.jit` decorators where appropriate
4. **Visualizations**: Use Plotly with `HTML(fig.to_html(...))` output
5. **Text formatting**: Avoid bold text (`**text**`) in markdown cells; use plain text or headers instead
6. **Capitalization**: Avoid all-caps words (use "states and controls" not "states AND controls")
7. **Heading case**: Use sentence case for section headings (use "Building problems" not "Building Problems")
8. **Adverbs**: Minimize adverb usage (avoid "particularly", "relatively", "significantly", "specifically", etc.)
9. **Comments**: Ensure all code comments are punctuated with periods

### Import Placement

Keep visualization imports (plotly, trimesh, IPython.display, etc.) in the visualization
cells where they're used, not at the top of the notebook. Since visualization cells
typically have the `hide-input` tag, this keeps the visible imports focused on the
core libraries (jax, jaxls, jaxlie).

```python
# Top import cell (visible) - core libraries only
import jax
import jax.numpy as jnp
import jaxlie
import jaxls

# Visualization cell (hidden with hide-input tag)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML

fig = go.Figure()
# ... visualization code ...
HTML(fig.to_html(...))
```

### Batched Cost/Variable Construction

When creating costs or variables, **prefer batched construction over Python loops**:

```python
# INEFFICIENT - Python loop (add comment if used for clarity)
costs = []
for i in range(n):
    costs.append(my_cost(MyVar(id=i), data[i]))
# Note: loop used for clarity; batched construction is more efficient

# EFFICIENT - Batched construction (preferred)
costs = [my_cost(MyVar(id=jnp.arange(n)), data)]
```

Batched construction:
- Passes arrays of IDs instead of scalar IDs
- Passes stacked data arrays instead of individual elements
- Results in vectorized cost evaluation
- Significantly faster for large problems

If using a Python loop for pedagogical clarity, add a comment noting that
batched construction would be more efficient in practice.

### Cell Metadata Tags

Use these tags in cell metadata to control display:
- `"hide-input"` - Hide the code, show only output
- `"hide-output"` - Hide output, show only code
- `"hide-cell"` - Hide entire cell

Example metadata:
```json
{
  "metadata": {
    "tags": ["hide-input"]
  }
}
```

Cells that should typically have `hide-input`:
- Logger configuration cells
- Large visualization/plotting code
- Helper function definitions

### API Cross-References

Use MyST syntax for linking to API docs:
```markdown
{class}`~jaxls.SE3Var`           # Short form: "SE3Var"
{class}`jaxls.SE3Var`            # Full form: "jaxls.SE3Var"
{func}`@jaxls.Cost.factory <jaxls.Cost.factory>`
```

## Custom CSS

Located in `source/_static/css/custom.css`:
- Limits height of text outputs (scrollable)
- Does NOT limit Plotly plot heights
- Styles copy button on outputs

## Adding a New Notebook

1. Create notebook in `docs/source/notebooks/`
2. Add to toctree in `docs/source/index.rst`
3. Follow style guidelines above
4. Test execution: `jupyter execute your_notebook.ipynb`
5. Build docs: `make html`

## Troubleshooting

### Notebook execution fails
- Check timeout isn't exceeded (increase in `conf.py` if needed)
- Ensure all imports are available
- Test notebook manually first

### Plots cut off
- CSS should not limit `.output_html` height
- Check `custom.css` for any conflicting styles

### Stale outputs
- Run `make clean && make html` for full rebuild
- Or delete `build/.jupyter_cache/`
