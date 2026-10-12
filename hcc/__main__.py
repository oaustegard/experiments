"""Entry point: `python3 -m hcc [gcc-like flags] files...`."""
import sys

from hcc.driver import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
