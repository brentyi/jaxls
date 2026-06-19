# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Quiet XLA's per-process C++ log spam (e.g. the harmless "GPU interconnect
# information not available / NVLink" warning on consumer GPUs) when the docs
# build executes notebooks on the GPU. 3 = errors only. Set before any notebook
# kernel is spawned and only as a default, so an explicit override still wins.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# -- Project information -----------------------------------------------------

project = "jaxls"
copyright = "2026"
author = "brentyi"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Autodoc settings
autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__init__, __post_init__",
}

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}
master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "default"

# MyST-NB settings
nb_execution_mode = "force"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
nb_merge_streams = True
# Execute notebooks serially. Parallel execution lets co-tenant notebooks
# contend for the GPU/CPU, which inflates any wall-clock the notebooks measure
# and report (e.g. the solver's verbose "solved in ..." line) and risks CUDA
# OOM when several heavy solves run at once.
nb_execution_parallel = 1

# MyST parser settings - enable math with dollar signs
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "jaxls"
html_static_path = ["_static"]
html_extra_path = [".nojekyll"]
templates_path = ["_templates"]

html_theme_options = {
    "repository_url": "https://github.com/brentyi/jaxls",
    "use_repository_button": True,
    "use_issues_button": True,
}

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_selector = "div.highlight pre"  # Include all code blocks including outputs


def setup(app):
    app.add_css_file("css/custom.css")
    app.add_js_file("js/custom.js")
