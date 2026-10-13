#include "region.h"
#include "mextend.h"
#include "rgn_add.h"

static void rgn_add(long key, long lo, long hi, long weight)
{
	int i, n = rgn_n;

	for (i = 0; i < n; i++) {
		if (rgn[i].key == key && rgn[i].lo < hi && rgn[i].hi > lo) {
			if (lo > rgn[i].lo)
				lo = rgn[i].lo;
			if (hi < rgn[i].hi)
				hi = rgn[i].hi;
			weight += rgn[i].weight;
			rgn[i].key = -1;
		}
	}
	for (i = 0; ; i++) {
		if (rgn[i].key < 0)
			break;
		if (i + 1 == n) {
			i = n;
			break;
		}
	}
	if (n == i) {
		if (rgn_sz <= n) {
			int sz = rgn_sz * 2;

			if (sz < 16)
				sz = 16;
			rgn_sz = sz;
			rgn = mextend(rgn, n, sz, sizeof(struct region));
		}
		rgn_n++;
	}
	rgn[i].key = key;
	rgn[i].lo = lo;
	rgn[i].hi = hi;
	rgn[i].weight = weight;
	rgn[i].reg = -1;
}
