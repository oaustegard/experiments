#!/bin/sh
# fp32b (instrumented rerun: beam counts, and a determinism check against fp32),
# then the n-gram arms on the pages the beam search touched, then weight arms.
R=$(dirname "$0")
E=/tmp/indra-eval
sh $R/sweep.sh fp32b
python3 $R/beam_pages.py $E/out/fp32b > $E/beam_pages.txt
chmod a+r $E/beam_pages.txt
sh $R/sweep.sh lm-off lm-o3 lm-top1 lm-q4 lm-q3 lm-o2 -- --pages beam_pages.txt --base out/fp32b
sh $R/sweep.sh s4c s3c s5c s4g32 s6c s3g32
