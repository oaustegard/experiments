set -e
cd /tmp/jail/tess534
export OMP_THREAD_LIMIT=1
for v in best fast; do
  mkdir -p work/out/time_${v}
  s=$(date +%s%N)
  harness/tess_dump work/models/$v/eng.traineddata work/out/time_${v} $(cat work/bench/eng.list) > /dev/null
  e=$(date +%s%N)
  echo "$v $(( (e - s) / 1000000 )) ms $(wc -l < work/bench/eng.list) lines"
done
