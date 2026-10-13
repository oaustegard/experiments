#include <stdlib.h>
#include <string.h>
#include "otf_chain.h"
#include "coverage.h"
#include "gctx_lookahead.h"
#include "mkfn_byname.h"
#include "str16.h"
#include "otf_gsubtype1.h"

#define U16(p) ({ unsigned short w_ = *(const unsigned short *)(p); (int)(unsigned short)((w_ << 8) | (w_ >> 8)); })

#define GNAME(g) \
	(!mkfn_byname && strlen(glyph_name[g]) > 3 \
	    ? (sprintf(str, "%d", (g)), str) : glyph_name[g])

static void
otf_gsubtype1(const unsigned char *t, struct otf_chain *c, int unused)
{
	int *g;
	int fmt, n, i, j, v, ng;

	fmt = ((t[0] << 8) | t[1]);
	g = coverage(t + U16(t + 2), &n);
	switch (fmt) {
	case 1:
		for (i = 0; i < n; i++) {
			v = g[i] + (short)U16(t + 4);
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
		break;
	case 2:
		for (i = 0; i < ((t[4] << 8) | t[5]); i++) {
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
		break;
	}
	free(g);
}
