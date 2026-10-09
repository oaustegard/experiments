#!/bin/sh
# Harness venv (pandas, pyarrow, swebench 4.0.4 for its log parsers), the
# interpreters the task envs need, and the repo mirrors. Idempotent.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=${SWE_LADDER_ROOT:-/tmp/jail/swe}
uvpy=${SWE_LADDER_UVPY:-/opt/uvpy}

[ -x "$here/.venv/bin/python" ] || uv venv -q -p 3.12 "$here/.venv"
uv pip install -q -p "$here/.venv/bin/python" 'swebench==4.0.4' pandas pyarrow

# Interpreters outside /root, which the jail's uid cannot enter.
UV_PYTHON_INSTALL_DIR=$uvpy uv python install --no-bin 3.8 3.9 3.10 3.11
chmod -R a+rX "$uvpy"

mkdir -p "$root/mirrors" "$root/envs" "$root/work" "$root/bin"
chmod 755 "$(dirname "$root")" "$root" "$root/mirrors" "$root/envs" "$root/work" "$root/bin"
for r in django/django sympy/sympy; do
  m="$root/mirrors/${r#*/}.git"
  [ -d "$m" ] || git clone -q --bare "https://github.com/$r" "$m"
  # Untracked files the harness puts in a checkout stay out of every diff.
  printf '.venv\n.swe-task.json\n__pycache__/\n*.pyc\n' > "$m/info/exclude"
done
"$here/.venv/bin/python" "$here/harness/fetch_tasks.py"

# swe-test / swe-run: the agents' jailed route to running task code.
for w in swe-test swe-run; do
  printf '#!/bin/sh\nexec python3 -I %s/harness/jailrun.py %s "$@"\n' "$here" "$w" > "$root/bin/$w"
  chmod 755 "$root/bin/$w"
done
