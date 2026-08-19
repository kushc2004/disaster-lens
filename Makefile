.PHONY: test smoke audit manifest

test:
	uv run pytest

smoke:
	uv run python scripts/create_smoke_bright.py

audit:
	uv run python scripts/inspect_bright.py data=bright

manifest:
	uv run python scripts/build_manifest.py data=bright

