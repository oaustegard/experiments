#include "mextend.h"
#include "lab_tab.h"
#include "ic_tab.h"

void o_label(long x)
{
	long *loc = lab_loc;
	long sz;

	while (x >= lab_sz) {
		sz = 2 * lab_sz;
		sz = sz < 0x80 ? 0x80 : sz;
		lab_sz = sz;
		loc = mextend(loc, lab_n, lab_sz, 8);
		lab_loc = loc;
	}
	long n = lab_n;
	if (x < n)
		goto store;
	memset(loc + n, 0xff, 8 * (x - n + 1));
	lab_n = x + 1;
store:
	loc[x] = ic_n;
	lab_last = ic_n;
}
