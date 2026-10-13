#include "font_gpos.h"
#include "font_readgpat.h"

static void font_readgpos(struct font *f, FILE *fp, int n, int a, int sel, int c, int d)
{
	struct frule *r;
	struct fent *e;
	char buf[128];
	char *p;
	int i;

	r = font_gpos(f, n, a, sel, c, d);
	for (i = 0; i < n; i++) {
		if (fscanf(fp, "%127s", buf) != 1)
			break;
		p = strchr(buf, ':');
		if (p)
			*p = 0;
		e = r->ent + i;
		e->flags = 1;
		if (buf[0] == 0)
			break;
		if (font_readgpat(f, e, buf))
			break;
		if (p)
			sscanf(p + 1, "%hd%hd%hd%hd", &e->aux, &e->a8, &e->aa, &e->ac);
	}
}
