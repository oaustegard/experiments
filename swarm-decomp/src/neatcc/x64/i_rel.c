#include "mextend.h"

static void i_rel(long sym, long type, long off)
{
	long sz;

	if (rel_n == rel_sz) {
		sz = 2 * rel_n;
		sz = sz < 0x80 ? 0x80 : sz;
		rel_sz = sz;
		rel_sym = mextend(rel_sym, rel_n, rel_sz, 8);
		rel_flg = mextend(rel_flg, rel_n, rel_sz, 8);
		rel_off = mextend(rel_off, rel_n, rel_sz, 8);
	}
	rel_sym[rel_n] = sym;
	rel_flg[rel_n] = type;
	rel_off[rel_n] = off;
	rel_n++;
}
