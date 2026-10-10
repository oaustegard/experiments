You are implementing the Python library `statsmodels` (upstream: statsmodels/statsmodels) from its skeleton.

The checkout at /tmp/jail/c0/work/s-swarm/statsmodels keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `statsmodels`. Make it pass by writing the code under `statsmodels/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `statsmodels/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test statsmodels -x -q` or `/tmp/jail/c0/bin/c0-test statsmodels/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import statsmodels'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

You are builder 11 of 19. All 19 builders work at the same time in this same checkout, each on its own files. Yours:
  - statsmodels/compat/python.py
  - statsmodels/datasets/fair/data.py
  - statsmodels/datasets/stackloss/__init__.py
  - statsmodels/datasets/strikes/data.py
  - statsmodels/distributions/bernstein.py
  - statsmodels/examples/ex_feasible_gls_het_0.py
  - statsmodels/examples/ex_nearest_corr.py
  - statsmodels/examples/example_ols_minimal_comp.py
  - statsmodels/examples/tsa/ex_var_reorder.py
  - statsmodels/gam/smooth_basis.py
  - statsmodels/graphics/dotplots.py
  - statsmodels/graphics/regressionplots.py
  - statsmodels/iolib/api.py
  - statsmodels/regression/_tools.py
  - statsmodels/sandbox/distributions/examples/ex_extras.py
  - statsmodels/sandbox/examples/example_gam_0.py
  - statsmodels/sandbox/gam.py
  - statsmodels/sandbox/nonparametric/smoothers.py
  - statsmodels/sandbox/regression/__init__.py
  - statsmodels/sandbox/regression/kernridgeregress_class.py
  - statsmodels/sandbox/stats/multicomp.py
  - statsmodels/sandbox/stats/runs.py
  - statsmodels/sandbox/tsa/fftarma.py
  - statsmodels/stats/inter_rater.py
  - statsmodels/stats/knockoff_regeffects.py
  - statsmodels/stats/oaxaca.py
  - statsmodels/tools/catadd.py
  - statsmodels/tsa/arima/datasets/brockwell_davis_2002/__init__.py
  - statsmodels/tsa/exponential_smoothing/base.py
  - statsmodels/tsa/innovations/api.py
  - statsmodels/tsa/stattools.py
  - statsmodels/tsa/vector_ar/hypothesis_test_results.py

Implement every stub in your files. Do not edit any other file: another builder owns it and is changing it right now. Code in other files may still be stubs while you work, so tests that need it can fail for reasons that are not yours; focus on making the tests that exercise your files pass, and when one fails because of another builder's file, move on. Run targeted tests (a test file or `-k` expression) rather than the whole suite most of the time.

This is a long job: expect a hundred or more tool calls. Do not stop to ask questions and do not report until every stub in your files is implemented and the tests that exercise them pass, apart from failures caused by other builders' files.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/s-swarm/out/statsmodels.w11.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
