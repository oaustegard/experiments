#include <stdlib.h>
#include <string.h>
#include "otf_chain.h"
#include "coverage.h"
#include "gctx_lookahead.h"
#include "mkfn_byname.h"
#include "str16.h"
#include "otf_gsubtype4.h"

#define U16(p) ({ unsigned short w_ = *(const unsigned short *)(p); (int)(unsigned short)((w_ << 8) | (w_ >> 8)); })

#define GNAME(g) \
	(!mkfn_byname && strlen(glyph_name[g]) > 3 \
	    ? (sprintf(str, "%d", (g)), str) : glyph_name[g])

static void
otf_gsubtype4(const unsigned char *t, struct otf_chain *c)
{
const unsigned char *set;
const unsigned char *lig;
int *g;
int i;
int k;
int j;
int nset;
int nlig;
int ncomp;


	if (U16(t) == 1) {
		g = coverage(t + U16(t + 2), NULL);
		nset = U16(t + 4);
		for (i = 0; i < nset; i++) {
			set = t + U16(t + 6 + 2 * i);
			nlig = U16(set);
			if (nlig == 0)
				continue;
			for (k = 0; k < nlig; k++) {
				lig = set + U16(set + 2 + 2 * k);
				ncomp = U16(lig + 2);
				if (c) {
					printf("%d", c->ni + c->nb + c->nl + 1);
					for (j = 0; j < c->nb; j++)
						printf(" =@%d", c->bt[j]);
					for (j = 0; j < c->seq; j++)
						printf(" =@%d", c->in[j]);
				} else
					printf("%d", ncomp + 1);
				printf(" -%s", GNAME(g[i]));
				for (j = 1; j < ncomp; j++)
					printf(" -%s", GNAME(U16(lig + 2 + 2 * j)));
				printf(" +%s", GNAME(U16(lig)));
				gctx_lookahead(c, ncomp);
				putchar('\n');
			}
		}
		free(g);
	}
}
