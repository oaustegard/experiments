#include "mextend.h"

static void lab_add(long n)
{
	long *loc = lab_loc;
	long sz;
	int old;

	while (n >= lab_sz) {
		old = lab_sz;
		sz = 2 * lab_sz;
		sz = sz < 0x80 ? 0x80 : sz;
		lab_sz = sz;
		loc = mextend(loc, old, sz, 8);
		lab_loc = loc;
	}
	loc[n] = mem_len(&cs);
}
