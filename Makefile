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

# DOCS_EXTRA selects the dependency set for the docs build. Default "docs" is
# CPU-only and portable; pass DOCS_EXTRA=docs-gpu to execute notebooks on the
# GPU (see build_and_deploy_docs.sh, which picks this automatically).
DOCS_EXTRA ?= docs

docs:
	uv run --extra $(DOCS_EXTRA) sphinx-build -b dirhtml docs/source docs/build/dirhtml

docs-clean:
	rm -rf docs/build
	$(MAKE) docs

benchmark:
	uv run --extra examples --extra docs python benchmark.py
