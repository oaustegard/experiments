#include <stdlib.h>
#include <string.h>
#include "otf_chain.h"
#include "coverage.h"
#include "gctx_lookahead.h"
#include "mkfn_byname.h"
#include "str16.h"
#include "otf_gsubtype3.h"

#define U16(p) ((int)__builtin_bswap16(*(const unsigned short *)(p)))

#define GNAME(g) \
	(!mkfn_byname && strlen(glyph_name[g]) > 3 \
	    ? (sprintf(str, "%d", (g)), str) : glyph_name[g])
static void
otf_gsubtype3(const unsigned char *t, struct otf_chain *c)
{
	const unsigned char *sub, *p;
	int *g;
	int i, j, n, cnt;

	if (U16(t) == 1) {
		g = coverage(t + U16(t + 2), NULL);
		n = U16(t + 4);
		for (i = 0; i != n; i++) {
			sub = t + U16(t + 6 + 2 * i);
			cnt = U16(sub);
			if (cnt == 0)
				continue;
			for (p = sub + 2; p != sub + 2 + 2 * cnt; p += 2) {
				if (c) {
					printf("%d", c->nb + c->ni + c->nl + 1);
					for (j = 0; j < c->nb; j++)
						printf(" =@%d", c->bt[j]);
					for (j = 0; j < c->seq; j++)
						printf(" =@%d", c->in[j]);
				} else
					printf("%d", 2);
				printf(" -%s", GNAME(g[i]));
				printf(" +%s", GNAME(U16(p)));
				gctx_lookahead(c, 1);
				putchar('\n');
			}
		}
		free(g);
	}
}
