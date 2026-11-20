# Dict-on (and dict-off) beam dumps for an arbitrary list of PNGs at model height.
# usage: sh harness/dump_dict_list.sh <variant> <lang> <list> <out_name>
set -e
cd /tmp/jail/tess534
v=$1; l=$2; list=$3; name=$4
mkdir -p work/out/dict/$name
harness/tess_dump_dict work/models/$v/$l.traineddata $l work/out/dict/$name $(cat $list) > work/out/dict/$name.tsv
echo "$name $(wc -l < work/out/dict/$name.tsv) lines"
