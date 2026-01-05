"""Tests for jaxls visualization utilities."""

import base64
from unittest.mock import MagicMock, patch

import jax
import jaxls


def test_show_jupyter():
    """Test that show() displays inline in Jupyter context."""
    vars = (jaxls.SE2Var(0),)

    @jaxls.Cost.factory
    def cost(vals: jaxls.VarValues, var: jaxls.SE2Var) -> jax.Array:
        return vals[var].log()

    problem = jaxls.LeastSquaresProblem([cost(vars[0])], vars)

    # Mock a Jupyter kernel environment.
    mock_ipython = MagicMock()
    mock_ipython.__class__.__name__ = "ZMQInteractiveShell"

    with (
        patch("IPython.get_ipython", return_value=mock_ipython),
        patch("IPython.display.display") as mock_display,
    ):
        result = problem.show()

        assert result is None
        assert mock_display.called
        iframe_obj = mock_display.call_args[0][0]
        # IFrame stores the src in the src attribute.
        data_uri = iframe_obj.src
        assert data_uri.startswith("data:text/html;base64,")
        # Decode and verify content.
        html_b64 = data_uri.split(",", 1)[1]
        html_str = base64.b64decode(html_b64).decode("utf-8")
        assert "<canvas>" in html_str
        assert "d3js.org" in html_str


def test_show_browser_fallback():
    """Test that show() opens browser when not in Jupyter."""
    vars = (jaxls.SE2Var(0),)

    @jaxls.Cost.factory
    def cost(vals: jaxls.VarValues, var: jaxls.SE2Var) -> jax.Array:
        return vals[var].log()

    problem = jaxls.LeastSquaresProblem([cost(vars[0])], vars)

    # Mock non-Jupyter environment (get_ipython returns None).
    with (
        patch("IPython.get_ipython", return_value=None),
        patch("webbrowser.open") as mock_browser,
    ):
        result = problem.show()

        assert result is None
        assert mock_browser.called
        url = mock_browser.call_args[0][0]
        assert url.startswith("file://")
        assert url.endswith(".html")
