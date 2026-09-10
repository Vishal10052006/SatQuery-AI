"""Small smoke-test/demo entry point for the M4 orchestration layer."""

from __future__ import annotations

import json

from .pipeline import run_agent


def main() -> None:
    """Run a representative SIH change-analysis query."""

    query = (
        "Has the built-up area increased between these two images? "
        "Verify it using optical and SAR data."
    )

    result = run_agent(query)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
