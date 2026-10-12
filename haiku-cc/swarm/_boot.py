"""Import bootstrap for entry scripts run as `python3 swarm/X.py`.

Python puts swarm/ first on sys.path for such scripts, and swarm/queue.py would
then shadow the stdlib `queue` (which concurrent.futures needs). Entry scripts
call this first so swarm modules import as `swarm.X` and the stdlib wins.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.dirname(HERE))
