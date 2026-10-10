You are implementing the Python library `statsmodels` (upstream: statsmodels/statsmodels) from its skeleton.

The checkout at /tmp/jail/c0/work/s-swarm/statsmodels keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `statsmodels`. Make it pass by writing the code under `statsmodels/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `statsmodels/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test statsmodels -x -q` or `/tmp/jail/c0/bin/c0-test statsmodels/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import statsmodels'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

You are builder 7 of 19. All 19 builders work at the same time in this same checkout, each on its own files. Yours:
  - statsmodels/base/_penalized.py
  - statsmodels/base/l1_slsqp.py
  - statsmodels/datasets/committee/__init__.py
  - statsmodels/datasets/nile/__init__.py
  - statsmodels/datasets/randhie/data.py
  - statsmodels/examples/ex_emplike_1.py
  - statsmodels/examples/ex_lowess.py
  - statsmodels/examples/example_discrete_mnl.py
  - statsmodels/examples/tsa/ar1cholesky.py
  - statsmodels/examples/tsa/compare_arma.py
  - statsmodels/genmod/bayes_mixed_glm.py
  - statsmodels/genmod/generalized_linear_model.py
  - statsmodels/graphics/tsaplots.py
  - statsmodels/graphics/tukeyplot.py
  - statsmodels/imputation/bayes_mi.py
  - statsmodels/miscmodels/tmodel.py
  - statsmodels/sandbox/archive/__init__.py
  - statsmodels/sandbox/distributions/multivariate.py
  - statsmodels/sandbox/distributions/quantize.py
  - statsmodels/sandbox/examples/ex_onewaygls.py
  - statsmodels/sandbox/nonparametric/__init__.py
  - statsmodels/sandbox/panel/mixed.py
  - statsmodels/sandbox/sysreg.py
  - statsmodels/sandbox/tools/cross_val.py
  - statsmodels/src/__init__.py
  - statsmodels/stats/moment_helpers.py
  - statsmodels/tsa/ardl/pss_critical_values.py
  - statsmodels/tsa/arima/estimators/innovations.py
  - statsmodels/tsa/arima/params.py
  - statsmodels/tsa/filters/api.py
  - statsmodels/tsa/filters/cf_filter.py
  - statsmodels/tsa/vector_ar/svar_model.py

Implement every stub in your files. Do not edit any other file: another builder owns it and is changing it right now. Code in other files may still be stubs while you work, so tests that need it can fail for reasons that are not yours; focus on making the tests that exercise your files pass, and when one fails because of another builder's file, move on. Run targeted tests (a test file or `-k` expression) rather than the whole suite most of the time.

This is a long job: expect a hundred or more tool calls. Do not stop to ask questions and do not report until every stub in your files is implemented and the tests that exercise them pass, apart from failures caused by other builders' files.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/s-swarm/out/statsmodels.w7.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
