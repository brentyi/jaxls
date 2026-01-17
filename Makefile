.PHONY: check lint format test transpile docs docs-clean benchmark

check:
	uv run --extra dev --extra examples pyright ./src ./examples ./tests

lint:
	uv run --extra dev --extra examples ruff check --fix .

format:
	uv run --extra dev --extra examples ruff format .

test:
	uv run --extra dev pytest tests/

transpile:
	uv run --extra dev ./transpile_py310.py

docs:
	uv run --extra docs sphinx-build -b dirhtml docs/source docs/build/dirhtml

docs-clean:
	rm -rf docs/build
	$(MAKE) docs

benchmark:
	uv run --extra examples --extra docs python benchmark.py
