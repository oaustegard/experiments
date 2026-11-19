set -e
cd /tmp/jail/tess534
model=$1; out=$2; shift 2
mkdir -p $out
harness/tess_dump $model $out "$@"
