#include "mkfn_res.h"

static int
uwid(int x)
{
	int u = upm;
	int v = x * 1000;
	int k = 72000 / mkfn_res;

	if (x < 0) {
		int h = u / 2;
		int r = (v - h) / u;
		return 10 * (r - k / 20) / k;
	}
	else {
		int h = u / 2;
		int r = (v + h) / u;
		return 10 * (r + k / 20) / k;
	}
}
