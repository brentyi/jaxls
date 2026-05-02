import subprocess
import sys


def test_import_with_modern_jax() -> None:
    """Guard against import-time use of JAX internals removed in JAX 0.6."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; import jaxls; print(jax.__version__, jaxls.__version__)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
