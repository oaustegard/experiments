set -e
cd /tmp/jail/tess534
mkdir -p $(dirname $2)
harness/tess_layers "$@"
