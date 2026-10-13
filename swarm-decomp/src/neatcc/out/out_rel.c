#include "fn_out_rel.h"
#include "reltab.h"

void
out_rel(long sym, long type, long off)
{
	struct rel *t;
	struct rel *r;
	long n;
	long k;
	if (type & 0x20)
		k = 2;
	else if (type & 0x800)
		k = 10 + ((type & 0x40) != 0);
	else
		k = 1;
	if (type & 2) {
		if (dsrel_n >= dsrel_sz) {
			long sz = dsrel_sz * 2;
			if (sz < 0x80)
				sz = 0x80;
			dsrel_sz = sz;
			dsrel = mextend(dsrel, dsrel_n, sz, sizeof(struct rel));
		}
		n = dsrel_n;
		dsrel_n = n + 1;
		t = dsrel;
	} else {
		if (csrel_n >= csrel_sz) {
			long sz = csrel_sz * 2;
			if (sz < 0x80)
				sz = 0x80;
			csrel_sz = sz;
			csrel = mextend(csrel, csrel_n, sz, sizeof(struct rel));
		}
		n = csrel_n;
		csrel_n = n + 1;
		t = csrel;
	}
	r = &t[n];
	r->off = off;
	r->info = (sym << 32) + k;
}
