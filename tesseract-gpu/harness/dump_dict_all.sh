# Dict-on (and dict-off) beam dumps for the 686 parity lines, every model in work/models.
# usage: sh harness/dump_dict_all.sh [variant ...]   (default: best fast)
set -e
cd /tmp/jail/tess534
vs="$*"; [ -n "$vs" ] || vs="best fast"
for v in $vs; do for l in eng fra grc; do
  mkdir -p work/out/dict/${v}_${l}
  s=$(date +%s%N)
  harness/tess_dump_dict work/models/$v/$l.traineddata $l work/out/dict/${v}_${l} $(cat work/bench/$l.list) > work/out/dict/${v}_${l}.tsv
  e=$(date +%s%N)
  echo "$v $l $(wc -l < work/out/dict/${v}_${l}.tsv) lines $(( (e-s)/1000000 )) ms"
done; done
