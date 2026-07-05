"""PyInstaller entry for the double-click GUI (windowed, no console)."""
import sys

from lbandscope.gui import main

if __name__ == "__main__":
    sys.exit(main())
