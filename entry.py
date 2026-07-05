"""PyInstaller entry shim -> single-file Windows binary."""
import sys

from lbandscope.cli import main

if __name__ == "__main__":
    sys.exit(main())
