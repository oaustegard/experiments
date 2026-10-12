"""Entry point: `python3 -m hcc [gcc-like flags] files...`.

Runs the driver in a thread with a large stack so deeply nested C
expressions don't overflow Python's recursion limit.
"""
import sys
import threading

from hcc.driver import main


def run() -> int:
    result = [70]

    def target():
        result[0] = main(sys.argv[1:])
    sys.setrecursionlimit(2000000)
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=target)
    t.start()
    t.join()
    return result[0]


if __name__ == "__main__":
    sys.exit(run())
