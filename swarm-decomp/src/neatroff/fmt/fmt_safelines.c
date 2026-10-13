#include "fn_nreg.h"
#include "fn_f_nexttrap.h"

static int fmt_safelines(void)
{
	int n, m, t;

	n = 1;
	if (*nreg('L') > 1)
		n = *nreg('L');
	m = *nreg('v');
	t = 1000;
	if (*nreg('v') > 0) {
		t = f_nexttrap();
		m *= n;
		return (m + t - 1) / m;
	}
	return t;
}
