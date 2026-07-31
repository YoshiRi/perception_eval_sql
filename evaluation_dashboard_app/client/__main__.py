"""``python -m client`` -> the CLI."""

from __future__ import annotations

import sys

from client.cli import main

if __name__ == "__main__":
    sys.exit(main())
