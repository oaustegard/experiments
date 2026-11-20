set -e
cd /tmp/jail/tess534
mkdir -p work/out/contrib_grc_hist
harness/tess_dump work/models/contrib/grc_hist.traineddata work/out/contrib_grc_hist $(cat work/bench/grc.list) > work/out/contrib_grc_hist.tsv
wc -l < work/out/contrib_grc_hist.tsv
