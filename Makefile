.PHONY: test audit manifest

test:
	uv run pytest

audit:
	uv run python scripts/inspect_bright.py data=bright

manifest:
	uv run python scripts/build_manifest.py data=bright
