#!/bin/sh
# The knobs that plausibly fix a greedy covering learner's overfit.
set -e
python3 fit.py --vocab schema --score laplace --min-cov 8  --tag laplace8
python3 fit.py --vocab schema --overlap 3                  --tag overlap
python3 fit.py --vocab schema --overlap 3 --score laplace --min-cov 8 --tag overlap-lap8
python3 fit.py --vocab schema --no-bigrams --score laplace --min-cov 8 --tag uni-lap8
echo SWEEPDONE
