# jaxls Development Guidelines

## Commands
- Install: `pip install -e ".[dev]"`
- Type checking: `pyright .`
- Linting: `ruff .`
- Run example: `python examples/pose_graph_simple.py`
- Run specific test: `python examples/test_cache.py`

## Code Style
- Use Python 3.10+ features
- Follow PEP 8 naming conventions: snake_case for functions/variables, PascalCase for classes
- Type annotations required for all functions and classes
- Import order: standard lib → third-party → local modules
- Use relative imports within the package
- Factor graph code uses dataclasses and functional style
- Error handling should use assertions for preconditions
- Documentation in docstrings should follow NumPy style
- Leverage JAX's functional programming paradigm
- Vectorize operations where possible for performance