#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.rope_observability import main  # noqa: E402


if __name__ == "__main__":
    main()
