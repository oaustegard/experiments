#include "ic_ent.h"
#include "mextend.h"

/* append an entry; op bits 0x107f0 push its index on iv */
static struct ic_ent *ic_put(long op, long a, long b, long c)
{
	struct ic_ent *e;
	long n, sz;

	if (ic_n == ic_sz) {
		sz = 2 * ic_n;
		sz = sz < 0x80 ? 0x80 : sz;
		ic_sz = sz;
		ic = mextend(ic, ic_n, sz, sizeof(struct ic_ent));
	}
	n = ic_n;
	ic_n = n + 1;
	e = &ic[n];
	e->op = op;
	e->a = a;
	e->b = b;
	e->c = c;
	if (op & 0x107f0) {
		iv[iv_n++] = n;
	}
	return e;
}
