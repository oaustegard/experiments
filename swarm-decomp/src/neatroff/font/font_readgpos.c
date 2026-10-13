#include "font_gpos.h"
#include "font_readgpat.h"

static void font_readgpos(struct font *f, FILE *fp, int *pn, int *pa, int *psel, int *pc, int *pd)
{
	struct frule *r;
	char buf[128];
	char *p;
	int i;
	int n = *pn;

	r = font_gpos(f, n, *pa, *psel, *pc, *pd);
	for (i = 0; i < n; i++) {
		if (fscanf(fp, "%127s", buf) != 1)
			break;
		p = strchr(buf, ':');
		if (p)
			*p = 0;
		r->ent[i].flags = 1;
		if (buf[0] == 0)
			break;
		if (font_readgpat(f, r->ent + i, buf))
			break;
		if (p)
			sscanf(p + 1, "%hd%hd%hd%hd", &r->ent[i].aux, &r->ent[i].a8, &r->ent[i].aa, &r->ent[i].ac);
	}
}
