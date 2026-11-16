"""Entry point for the desktop application (also used by PyInstaller)."""
import sys

from cgaaf.gui import run

if __name__ == "__main__":
    sys.exit(run())
