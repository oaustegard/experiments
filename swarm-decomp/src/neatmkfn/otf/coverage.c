#include <stdlib.h>
#include "coverage.h"

#define U16(p) ((int)__builtin_bswap16(*(const unsigned short *)(p)))

static int *
coverage(const unsigned char *t, int *n)
{
	int *g;
	int fmt, cnt, i, k, s, e, v, m;

	fmt = U16(t);
	cnt = U16(t + 2);
	g = malloc(glyph_n * sizeof *g);
	m = 0;
	if (fmt == 1) {
		for (i = 0; i < cnt; i++)
			g[i] = U16(t + 4 + 2 * i);
		m = cnt;
	}
	if (fmt == 2) {
		for (k = 0; k < cnt; k++) {
			s = U16(t + 4 + 6 * k);
			e = U16(t + 6 + 6 * k);
			for (v = s; v <= e; v++)
				g[m++] = v;
		}
	}
	if (n)
		*n = m;
	return g;
}
