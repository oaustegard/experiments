You are implementing the Python library `statsmodels` (upstream: statsmodels/statsmodels) from its skeleton.

The checkout at /tmp/jail/c0/work/s-swarm/statsmodels keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `statsmodels`. Make it pass by writing the code under `statsmodels/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `statsmodels/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test statsmodels -x -q` or `/tmp/jail/c0/bin/c0-test statsmodels/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import statsmodels'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

You are builder 18 of 19. All 19 builders work at the same time in this same checkout, each on its own files. Yours:
  - statsmodels/base/l1_solvers_common.py
  - statsmodels/compat/patsy.py
  - statsmodels/datasets/ccard/data.py
  - statsmodels/datasets/fair/__init__.py
  - statsmodels/datasets/statecrime/data.py
  - statsmodels/distributions/copula/__init__.py
  - statsmodels/emplike/elregress.py
  - statsmodels/examples/ex_kernel_regression3.py
  - statsmodels/examples/ex_regressionplots.py
  - statsmodels/examples/try_fit_constrained.py
  - statsmodels/gam/gam_cross_validation/__init__.py
  - statsmodels/genmod/generalized_estimating_equations.py
  - statsmodels/graphics/functional.py
  - statsmodels/multivariate/factor_rotation/__init__.py
  - statsmodels/regression/feasible_gls.py
  - statsmodels/sandbox/distributions/extras.py
  - statsmodels/sandbox/distributions/transformed.py
  - statsmodels/sandbox/examples/thirdparty/findow_1.py
  - statsmodels/sandbox/tools/__init__.py
  - statsmodels/sandbox/tsa/varma.py
  - statsmodels/stats/dist_dependence_measures.py
  - statsmodels/tools/grouputils.py
  - statsmodels/tools/validation/decorators.py
  - statsmodels/tsa/ardl/model.py
  - statsmodels/tsa/arima/estimators/burg.py
  - statsmodels/tsa/arima_model.py
  - statsmodels/tsa/forecasting/stl.py
  - statsmodels/tsa/holtwinters/_smoothers.py
  - statsmodels/tsa/holtwinters/model.py
  - statsmodels/tsa/statespace/_smoothers/__init__.py
  - statsmodels/tsa/statespace/cfa_simulation_smoother.py
  - statsmodels/tsa/statespace/kalman_filter.py
  - statsmodels/tsa/statespace/varmax.py

Implement every stub in your files. Do not edit any other file: another builder owns it and is changing it right now. Code in other files may still be stubs while you work, so tests that need it can fail for reasons that are not yours; focus on making the tests that exercise your files pass, and when one fails because of another builder's file, move on. Run targeted tests (a test file or `-k` expression) rather than the whole suite most of the time.

This is a long job: expect a hundred or more tool calls. Do not stop to ask questions and do not report until every stub in your files is implemented and the tests that exercise them pass, apart from failures caused by other builders' files.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/s-swarm/out/statsmodels.w18.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
