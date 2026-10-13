#include "mkfn_res.h"

static int
uwid(int x)
{
	int u = upm;
	int k = 72000 / mkfn_res;

	if (x < 0) {
		int v = x * 1000;
		int r = (v - u / 2) / u;
		return 10 * (r - k / 20) / k;
	}
	else {
		int v = x * 1000;
		int r = (v + u / 2) / u;
		return 10 * (r + k / 20) / k;
	}
}
