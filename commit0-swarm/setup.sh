#!/bin/sh
# Harness venv, interpreters, and bare mirrors of the 16 Commit0-lite repos. Idempotent.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=${C0_ROOT:-/tmp/jail/c0}
uvpy=${C0_UVPY:-/opt/uvpy}

[ -x "$here/.venv/bin/python" ] || uv venv -q -p 3.12 "$here/.venv"
uv pip install -q -p "$here/.venv/bin/python" pandas pyarrow

UV_PYTHON_INSTALL_DIR=$uvpy uv python install --no-bin 3.10 3.12
chmod -R a+rX "$uvpy"

mkdir -p "$root/mirrors" "$root/envs" "$root/work" "$root/bin"
chmod 755 "$(dirname "$root")" "$root" "$root/mirrors" "$root/envs" "$root/work" "$root/bin"
for l in $("$here/.venv/bin/python" "$here/harness/common.py" names); do
  m="$root/mirrors/$l.git"
  [ -d "$m" ] || git clone -q --bare "https://github.com/commit-0/$l" "$m"
  printf '.venv\n.c0-task.json\n__pycache__/\n*.pyc\n.pytest_cache/\n' > "$m/info/exclude"
done

for w in c0-test c0-run; do
  printf '#!/bin/sh\nexec python3 -I %s/harness/jailrun.py %s "$@"\n' "$here" "$w" > "$root/bin/$w"
  chmod 755 "$root/bin/$w"
done
