BODIES_4 = [
# 1 _autolag
'''    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    elif method == "t-stat":
        stop = stats.norm.ppf(0.95)
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                break
    else:
        raise ValueError(f"Information criterion {method} not understood.")

    if not regresults:
        return icbest, bestlag
    return icbest, bestlag, results''',
# 2 adfuller
'''    x = array_like(x, "x")
    maxlag = int_like(maxlag, "maxlag", optional=True)
    store = bool_like(store, "store")
    regresults = bool_like(regresults, "regresults")
    if regression not in ("c", "ct", "ctt", "n"):
        raise ValueError(f"regression option {regression} not understood")
    if autolag is not None:
        autolag = autolag.lower()
        if autolag not in ("aic", "bic", "t-stat"):
            raise ValueError(f"autolag option {autolag} not understood")

    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")

    if regresults:
        store = True

    ntrend = len(regression) if regression != "n" else 0
    nobs = x.shape[0]

    if maxlag is None:
        # from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError("sample size is too short to use selected "
                             "regression component")
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError("maxlag must be less than (nobs/2 - 1 - ntrend) "
                         "where n trend is the number of included "
                         "deterministic regressors")
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1:-1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        resstore = ResultsStore()

    if autolag:
        if regression != "n":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1  # 1 for level
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        if not regresults:
            icbest, bestlag = _autolag(OLS, xdshort, fullRHS, startlag,
                                       maxlag, autolag)
        else:
            icbest, bestlag, alres = _autolag(OLS, xdshort, fullRHS, startlag,
                                              maxlag, autolag,
                                              regresults=regresults)
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim="both", original="in")
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1:-1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != "n":
        resols = OLS(xdshort, add_trend(xdall[:, :usedlag + 1],
                                        regression)).fit()
    else:
        resols = OLS(xdshort, xdall[:, :usedlag + 1]).fit()

    adfstat = resols.tvalues[0]
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {"1%": critvalues[0], "5%": critvalues[1],
                  "10%": critvalues[2]}
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = ("The coefficient on the lagged level equals 1 - "
                       "unit root")
        resstore.HA = "No unit root"
        resstore.icbest = icbest
        resstore.regression = regression
        resstore.autolag = autolag
        return adfstat, pvalue, usedlag, nobs, critvalues, icbest, resstore
    return adfstat, pvalue, usedlag, nobs, critvalues, icbest''',
# 3 acovf
'''    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d")
    if missing == "raise" and np.isnan(x).any():
        raise MissingDataError("NaN values found in x")
    if missing == "drop":
        x = x[~np.isnan(x)]
    n = x.shape[0]
    if nlag is None:
        nlag = n - 1
    if demean:
        xo = x - (np.nanmean(x) if missing == "conservative" else x.mean())
    else:
        xo = x.copy()
    xo = np.nan_to_num(xo) if missing == "conservative" else xo
    if fft:
        n_fft = _next_regular(2 * n + 1)
        Frf = np.fft.fft(xo, n=n_fft)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:n].real
    else:
        acov = np.array([np.dot(xo[:n - i], xo[i:]) for i in range(n)])
    acov = acov[:nlag + 1]
    if adjusted:
        acov = acov / (n - np.arange(len(acov)))
    else:
        acov = acov / n
    return acov''',
# 4 q_stat
'''    x = np.asarray(x)
    ret = nobs * (nobs + 2) * np.cumsum(
        (1.0 / (nobs - np.arange(1, len(x) + 1))) * x ** 2)
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return ret, chi2''',
# 5 acf
'''    nobs = len(x)
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft,
                missing=missing, nlag=nlags)
    acf = avf[:nlags + 1] / avf[0]
    if alpha is not None:
        if bartlett_confint:
            varacf = np.ones(nlags + 1) / nobs
            varacf[0] = 0
            varacf[1] = 1.0 / nobs
            varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
        else:
            varacf = 1.0 / nobs
        interval = stats.norm.ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval, acf + interval))
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)
        if alpha is not None:
            return acf, confint, qstat, pvalue
        return acf, qstat, pvalue
    if alpha is not None:
        return acf, confint
    return acf''',
# 6 pacf_yw
'''    if nlags is None:
        nlags = min(int(10 * np.log10(x.shape[0])), x.shape[0] // 2 - 1)
    pacf = [1.0]
    for k in range(1, nlags + 1):
        pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)''',
# 7 pacf_burg
'''    from statsmodels.regression.linear_model import burg

    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    pacf = [1.0]
    sigma2 = [np.dot(x - x.mean(), x - x.mean()) / nobs if demean
              else np.dot(x, x) / nobs]
    for k in range(1, nlags + 1):
        phi, sig2 = burg(x, order=k, demean=demean)
        pacf.append(phi[-1])
        sigma2.append(sig2)
    return np.array(pacf), np.array(sigma2)''',
# 8 pacf_ols
'''    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    if efficient:
        for k in range(1, nlags + 1):
            xlags, x0 = lagmat(x, k, trim="both", original="sep")
            xlags = add_constant(xlags)
            params = lstsq(xlags, x0)[0]
            pacf[k] = params[-1]
    else:
        xd = x - x.mean()
        xlags, x0 = lagmat(xd, nlags, trim="both", original="sep")
        for k in range(1, nlags + 1):
            params = lstsq(xlags[:, :k], x0)[0]
            pacf[k] = params[-1]
    if adjusted:
        pacf *= nobs / (nobs - np.arange(nlags + 1))
    return pacf''',
# 9 pacf
'''    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    if method == "ols":
        ret = pacf_ols(x, nlags=nlags, efficient=True, adjusted=False)
    elif method == "ols-inefficient":
        ret = pacf_ols(x, nlags=nlags, efficient=False, adjusted=False)
    elif method == "ols-adjusted":
        ret = pacf_ols(x, nlags=nlags, efficient=True, adjusted=True)
    elif method in ("yw", "ywm", "ywmle"):
        ret = pacf_yw(x, nlags=nlags, method="mle")
    elif method == "ywadjusted":
        ret = pacf_yw(x, nlags=nlags, method="adjusted")
    elif method in ("ld", "ldb", "ldbiased"):
        acv = acovf(x, adjusted=False, fft=False)
        ret = levinson_durbin(acv, nlags=nlags, isacov=True)[2]
    elif method == "ldadjusted":
        acv = acovf(x, adjusted=True, fft=False)
        ret = levinson_durbin(acv, nlags=nlags, isacov=True)[2]
    elif method == "burg":
        ret = pacf_burg(x, nlags=nlags, demean=True)[0]
    else:
        raise ValueError(f"method {method} not understood")
    if alpha is not None:
        varacf = 1.0 / len(x)  # for all lags >=1
        interval = stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(ret - interval, ret + interval))
        confint[0] = ret[0]
        return ret, confint
    return ret''',
# 10 ccovf
'''    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    cov = correlate(xo, yo, mode="full",
                    method="fft" if fft else "direct")[n - 1:]
    if adjusted:
        cov = cov / (n - np.arange(n))
    else:
        cov = cov / n
    return cov''',
# 11 ccf
'''    cvf = ccovf(x, y, adjusted=adjusted, demean=True, fft=fft)
    ccf = cvf / (np.std(x) * np.std(y))
    if nlags is not None:
        ccf = ccf[:nlags + 1]
    if alpha is not None:
        interval = stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(len(x))
        confint = np.array(lzip(ccf - interval, ccf + interval))
        return ccf, confint
    return ccf''',
# 12 levinson_durbin
'''    s = np.asarray(s)
    order = nlags
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[:order + 1]
    phi = np.zeros((order + 1, order + 1), "d")
    sig = np.zeros(order + 1)
    # initial points
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[0] = sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k - 1],
                                       sxx_m[1:k][::-1])) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)
    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.0
    return sigma_v, arcoefs, pacf_, sig, phi''',
# 13 levinson_durbin_pacf
'''    pacf = np.asarray(pacf)
    if nlags is None:
        nlags = len(pacf) - 1
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = pacf[1]
    for k in range(2, nlags + 1):
        phi[k, k] = pacf[k]
        phi[1:k, k] = phi[1:k, k - 1] - phi[k, k] * phi[k - 1:0:-1, k - 1]
    arcoefs = phi[1:, nlags]
    # autocorrelations implied by the Durbin recursion
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0
    acf[1] = pacf[1]
    for k in range(2, nlags + 1):
        s1 = np.dot(phi[1:k, k - 1], acf[1:k])
        s2 = np.dot(phi[1:k, k - 1], acf[k - 1:0:-1])
        acf[k] = pacf[k] * (1 - s1) + s2
    return arcoefs, acf''',
# 14 breakvar
'''    x = array_like(resid, "resid")
    nobs = x.shape[0]
    if isinstance(subset_length, float):
        h = int(subset_length * nobs)
    else:
        h = int(subset_length)
    squares = x ** 2
    numer = np.sum(squares[-h:])
    denom = np.sum(squares[:h])
    test_statistic = numer / denom
    if use_f:
        dist = stats.f(h, h)
    else:
        test_statistic = h * test_statistic
        dist = stats.chi2(h)
    if alternative == "increasing":
        p_value = dist.sf(test_statistic)
    elif alternative == "decreasing":
        p_value = dist.cdf(test_statistic)
    elif alternative == "two-sided":
        p_value = 2 * np.minimum(dist.cdf(test_statistic),
                                 dist.sf(test_statistic))
    else:
        raise ValueError("alternative must be increasing, decreasing or "
                         "two-sided")
    return test_statistic, p_value''',
# 15 grangercausalitytests
'''    x = array_like(x, "x", ndim=2)
    if x.shape[1] != 2:
        raise ValueError("x must have 2 columns")
    addconst = bool_like(addconst, "addconst")
    if isinstance(maxlag, (int, np.integer)):
        lags = np.arange(1, int(maxlag) + 1)
    else:
        lags = np.asarray(maxlag, dtype=int)
    if x.shape[0] <= 3 * lags.max() + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {}".format(int((x.shape[0] - int(addconst)) / 3) - 1))

    results = {}
    for mlg in lags:
        result = {}
        dta = lagmat2ds(x, mlg, trim="both", dropex=1)
        if addconst:
            dtaown = add_trend(dta[:, 1:mlg + 1], trend="c", prepend=False)
            dtajoint = add_trend(dta[:, 1:], trend="c", prepend=False)
        else:
            dtaown = dta[:, 1:mlg + 1]
            dtajoint = dta[:, 1:]
        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # Granger causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) / res2djoint.ssr
                * res2djoint.df_resid / mlg)
        fpval = stats.f.sf(fgc1, mlg, res2djoint.df_resid)
        result["ssr_ftest"] = (fgc1, fpval, res2djoint.df_resid, mlg)

        # Granger causality test using ssr (chi2 statistic)
        fgc2 = res2djoint.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        chi2pval = stats.chi2.sf(fgc2, mlg)
        result["ssr_chi2test"] = (fgc2, chi2pval, mlg)

        # likelihood ratio test
        lr = -2 * (res2down.llf - res2djoint.llf)
        lrpval = stats.chi2.sf(lr, mlg)
        result["lrtest"] = (lr, lrpval, mlg)

        # parameter F test: coefficients of lags of the second series are 0
        k_joint = dtajoint.shape[1]
        restriction = np.zeros((mlg, k_joint))
        restriction[:, mlg:2 * mlg] = np.eye(mlg)
        wald = res2djoint.wald_test(restriction, use_f=True)
        result["params_ftest"] = (float(np.squeeze(wald.statistic)),
                                  float(np.squeeze(wald.pvalue)),
                                  res2djoint.df_resid, mlg)
        results[mlg] = (result, [res2djoint, res2down, dtajoint])

    if verbose:
        for mlg in lags:
            print(f"\\nGranger Causality\\nnumber of lags (no zero) {mlg}")
            for name, vals in results[mlg][0].items():
                print(f"{name}: F/Chi2={vals[0]:.4f}, p={vals[1]:.4g}")
    return results''',
# 16 coint
'''    if trend not in ["n", "c", "ct"]:
        raise ValueError(f"trend option {trend} not understood")
    y0 = array_like(y0, "y0")
    y1 = array_like(y1, "y1", ndim=2)
    nobs, k_vars = y1.shape
    k_vars += 1  # add 1 for y0

    if trend == "n":
        xx = y1
    else:
        xx = add_trend(y1, trend=trend, prepend=False)
    res_co = OLS(y0, xx).fit()

    if res_co.rsquared < 1 - 100 * SQRTEPS:
        res_adf = adfuller(res_co.resid, maxlag=maxlag, autolag=autolag,
                            regression="n")
    else:
        warnings.warn("y0 and y1 are (almost) perfectly colinear."
                      "Cointegration test is not reliable in this case.",
                      CollinearityWarning, stacklevel=2)
        # Edge case where perfect collinearity happens
        # return the t-stat of -inf and pvalue of zero
        res_adf = (-np.inf, 0.0, None, None, None, None)

    coint_t = res_adf[0]
    pvalue = mackinnonp(coint_t, regression=trend, N=k_vars)
    crit = mackinnoncrit(N=k_vars, regression=trend, nobs=nobs)
    crit_value = {"1%": crit[0], "5%": crit[1], "10%": crit[2]}
    return coint_t, pvalue, crit_value''',
# 17 arma_order_select_ic
'''    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.tools import Bunch

    y = array_like(y, "y1")
    model_kw = {} if model_kw is None else model_kw
    fit_kw = {} if fit_kw is None else fit_kw
    if isinstance(ic, str):
        ic = [ic]
    ic = [i.lower() for i in ic]
    ar_range = range(max_ar + 1)
    ma_range = range(max_ma + 1)
    tables = {i: np.full((max_ar + 1, max_ma + 1), np.nan) for i in ic}
    for ar in ar_range:
        for ma in ma_range:
            try:
                res = ARIMA(y, order=(ar, 0, ma), trend=trend,
                            **model_kw).fit(**fit_kw)
            except (LinAlgError, ValueError):
                continue
            for i in ic:
                tables[i][ar, ma] = getattr(res, i)
    results = Bunch()
    for i in ic:
        table = pd.DataFrame(tables[i], index=list(ar_range),
                             columns=list(ma_range))
        results[i] = table
        if np.isnan(tables[i]).all():
            results[f"{i}_min_order"] = None
        else:
            idx = np.nanargmin(tables[i])
            results[f"{i}_min_order"] = (int(idx // (max_ma + 1)),
                                         int(idx % (max_ma + 1)))
    return results''',
# 18 has_missing
'''    return bool(np.isnan(np.sum(data)))''',
# 19 kpss
'''    if regression not in ["c", "ct"]:
        raise ValueError(f"regression option {regression} not understood")
    x = array_like(x, "x")
    nobs = x.shape[0]
    if regression == "c":
        resids = x - x.mean()
        crit = [0.347, 0.463, 0.574, 0.739]
    else:
        trend = np.arange(1, nobs + 1)
        xtrend = add_constant(trend)
        resids = OLS(x, xtrend).fit().resid
        crit = [0.119, 0.146, 0.176, 0.216]

    if nlags == "legacy":
        nlags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        nlags = min(nlags, nobs - 1)
    elif nlags == "auto":
        nlags = _kpss_autolag(resids, nobs)
        nlags = min(nlags, nobs - 1)
    else:
        nlags = int_like(nlags, "nlags")
        if nlags >= nobs:
            raise ValueError("lags must be less than the number of observations")

    pvals = [0.10, 0.05, 0.025, 0.01]
    eta = np.sum(np.cumsum(resids) ** 2) / (nobs ** 2)
    s_hat = _sigma_est_kpss(resids, nobs, nlags)
    kpss_stat = eta / s_hat
    if kpss_stat >= crit[-1]:
        warnings.warn("The test statistic is outside of the range of p-values"
                      " available in the look-up table. The actual p-value is"
                      " smaller than the p-value returned.",
                      InterpolationWarning, stacklevel=2)
        p_value = pvals[-1]
    elif kpss_stat <= crit[0]:
        warnings.warn("The test statistic is outside of the range of p-values"
                      " available in the look-up table. The actual p-value is"
                      " greater than the p-value returned.",
                      InterpolationWarning, stacklevel=2)
        p_value = pvals[0]
    else:
        p_value = np.interp(kpss_stat, crit, pvals)
    crit_dict = {"10%": crit[0], "5%": crit[1], "2.5%": crit[2], "1%": crit[3]}
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        resstore = ResultsStore()
        resstore.lags = nlags
        resstore.nobs = nobs
        return kpss_stat, p_value, nlags, crit_dict, resstore
    return kpss_stat, p_value, nlags, crit_dict''',
# 20 _sigma_est_kpss
'''    s_hat = np.sum(resids ** 2)
    for i in range(1, lags + 1):
        resids_prod = np.dot(resids[i:], resids[:-i])
        s_hat += 2 * resids_prod * (1.0 - (i / (lags + 1.0)))
    return s_hat / nobs''',
# 21 _kpss_autolag
'''    covlags = int(np.power(nobs, 2.0 / 9.0))
    s0 = np.sum(resids ** 2) / nobs
    s1 = 0.0
    for i in range(1, covlags + 1):
        resids_prod = np.dot(resids[i:], resids[:-i]) / nobs
        s0 += 2 * resids_prod
        s1 += 2 * i * resids_prod
    # Hobijn et al. (1998) bandwidth for the Bartlett kernel
    alpha1 = (s1 / s0) ** 2 if s0 != 0 else 0.0
    return int(1.1447 * np.power(alpha1 * nobs, 1 / 3.0))''',
# 22 range_unit_root_test
'''    x = array_like(x, "x")
    nobs = x.shape[0]
    dm = x - x.mean()
    cs = np.cumsum(dm)
    cs = np.concatenate(([0.0], cs))
    rur_stat = (cs.max() - cs.min()) / np.sqrt(np.sum(dm ** 2))
    rur_stat = rur_stat * np.sqrt(nobs)
    crit = {"1%": 0.3520, "5%": 0.2300, "10%": 0.1800}
    pvals = [0.10, 0.05, 0.01]
    crit_vals = [crit["10%"], crit["5%"], crit["1%"]]
    p_value = np.interp(rur_stat, crit_vals, pvals)
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        resstore = ResultsStore()
        resstore.nobs = nobs
        return rur_stat, p_value, crit, resstore
    return rur_stat, p_value, crit''',
]

BODIES_8 = [
# _za_crit
'''        table = self._za_critical_values[model]
        pvalues = table[:, 0]
        values = table[:, 1]
        pvalue = np.interp(stat, values, pvalues)
        cvdict = {
            "1%": np.interp(0.01, pvalues, values),
            "5%": np.interp(0.05, pvalues, values),
            "10%": np.interp(0.10, pvalues, values),
        }
        return pvalue, cvdict''',
# _quick_ols
'''        params = np.linalg.lstsq(exog, endog, rcond=None)[0]
        resid = endog - exog.dot(params)
        nobs, k = exog.shape
        sigma2 = resid.dot(resid) / max(nobs - k, 1)
        cov = sigma2 * np.linalg.pinv(exog.T.dot(exog))
        return params, np.sqrt(np.diag(cov))''',
# _format_regression_data
'''        dy = np.diff(series)
        n = len(dy)
        # lagged level and lagged differences, aligned to dy[lags:]
        level = series[lags:n]
        endog = dy[lags:]
        cols_list = [level[:, None]]
        for j in range(1, cols + 1):
            cols_list.append(dy[lags - j:n - j, None])
        exog = np.column_stack(cols_list)
        if const:
            exog = np.column_stack((np.ones(len(endog)), exog))
        if trend:
            exog = np.column_stack((exog, np.arange(1, len(endog) + 1)))
        return endog, exog''',
# _update_regression_exog
'''        # append the break dummies for a break at x index period
        t_idx = np.arange(exog.shape[0]) + lags + 1
        parts = [exog]
        if regression in ("c", "ct"):
            parts.append((t_idx >= period).astype(float)[:, None])
        if regression in ("t", "ct"):
            parts.append(np.maximum(t_idx - period, 0).astype(float)[:, None])
        return np.column_stack(parts)''',
# run
'''        x = array_like(x, "x")
        if regression not in ("c", "t", "ct"):
            raise ValueError(f"regression option {regression} not understood")
        trim = float_like(trim, "trim")
        if not 0 <= trim <= 1 / 3:
            raise ValueError("trim must be in [0, 1/3]")
        nobs = x.shape[0]
        if maxlag is None:
            maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
            maxlag = min(nobs // 2 - 1, maxlag)
        const = regression in ("c", "ct")
        trend = regression in ("t", "ct")

        # base model without break to select the lag length
        def base_design(k):
            return self._format_regression_data(x, nobs, const, trend, k,
                                                maxlag)

        if autolag is None:
            baselag = maxlag
        else:
            autolag = autolag.lower()
            best = None
            for k in range(maxlag + 1):
                endog_, exog_ = base_design(k)
                params, bse = self._quick_ols(endog_, exog_)
                ssr = np.sum((endog_ - exog_.dot(params)) ** 2)
                nobs_k = len(endog_)
                k_params = exog_.shape[1]
                if autolag == "aic":
                    ic = np.log(ssr / nobs_k) + 2.0 * k_params / nobs_k
                elif autolag == "bic":
                    ic = np.log(ssr / nobs_k) + k_params * np.log(nobs_k) / nobs_k
                else:
                    ic = np.abs(params[-1] / bse[-1]) if k > 0 else np.inf
                if best is None or (autolag != "t-stat" and ic < best[0]):
                    best = (ic, k)
            if autolag == "t-stat":
                baselag = maxlag
                for k in range(maxlag, 0, -1):
                    endog_, exog_ = base_design(k)
                    params, bse = self._quick_ols(endog_, exog_)
                    if np.abs(params[-1] / bse[-1]) >= 1.6448536269514722:
                        baselag = k
                        break
                else:
                    baselag = 0
            else:
                baselag = best[1]

        # break-period search over the trimmed sample
        endog_, exog_full = self._format_regression_data(
            x, nobs, const, trend, baselag, maxlag)
        start = int(np.floor(trim * nobs))
        end = nobs - start
        tstats = []
        periods = []
        # the lagged level is the first regressor after the constant
        level_pos = 1 if const else 0
        for tb in range(start, end):
            exog_b = self._update_regression_exog(exog_full, regression, tb,
                                                  nobs, const, trend,
                                                  baselag, maxlag)
            params, bse = self._quick_ols(endog_, exog_b)
            tstats.append(params[level_pos] / bse[level_pos])
            periods.append(tb)
        tstats = np.asarray(tstats)
        imin = int(np.argmin(tstats))
        zastat = tstats[imin]
        bpidx = periods[imin]
        pvalue, cvdict = self._za_crit(zastat, model=regression)
        return zastat, pvalue, cvdict, baselag, bpidx''',
]
