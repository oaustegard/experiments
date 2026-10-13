#include <stdlib.h>
#include <string.h>
#include "otf_chain.h"
#include "coverage.h"
#include "gctx_lookahead.h"
#include "mkfn_byname.h"
#include "str16.h"
#include "otf_gsubtype1.h"

#define U16(p) ((int)__builtin_bswap16(*(const unsigned short *)(p)))

#define GNAME(g) \
	(!mkfn_byname && strlen(glyph_name[g]) > 3 \
	    ? (sprintf(str, "%d", (g)), str) : glyph_name[g])

static void
otf_gsubtype1(const unsigned char *t, struct otf_chain *c)
{
	int *g;
	int fmt, n, i, j, v, ng;

	fmt = U16(t);
	g = coverage(t + U16(t + 2), &n);
	if (fmt == 1) {
		int delta = (short)U16(t + 4);

		for (i = 0; i < n; i++) {
			v = g[i] + delta;
			if (v >= glyph_n || v < 0)
				continue;
			if (c) {
				printf("%d", c->ni + c->nb + c->nl + 1);
				for (j = 0; j < c->nb; j++)
					printf(" =@%d", c->bt[j]);
				for (j = 0; j < c->seq; j++)
					printf(" =@%d", c->in[j]);
			} else
				printf("%d", 2);
			printf(" -%s", GNAME(g[i]));
			printf(" +%s", GNAME(v));
			gctx_lookahead(c, 1);
			putchar('\n');
		}
	} else if (fmt == 2) {
		ng = U16(t + 4);
		for (i = 0; i < ng; i++) {
			if (c) {
				printf("%d", c->ni + c->nb + c->nl + 1);
				for (j = 0; j < c->nb; j++)
					printf(" =@%d", c->bt[j]);
				for (j = 0; j < c->seq; j++)
					printf(" =@%d", c->in[j]);
			} else
				printf("%d", 2);
			printf(" -%s", GNAME(g[i]));
			printf(" +%s", GNAME(U16(t + 6 + 2 * i)));
			gctx_lookahead(c, 1);
			putchar('\n');
		}
	}
	free(g);
}
