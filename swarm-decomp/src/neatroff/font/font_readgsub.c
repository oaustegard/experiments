#include "font_gsub.h"
#include "font_readgpat.h"

static void font_readgsub(struct font *f, FILE *fp, int *pn, int *pa, int *psel, int *pc, int *pd)
{
	struct frule *r;
	char buf[128];
	int i;
	int n = *pn;

	r = font_gsub(f, n, *pa, *psel, *pc, *pd);
	for (i = 0; i < n; i++) {
		if (fscanf(fp, "%127s", buf) != 1)
			break;
		if (buf[0] == '-')
			r->ent[i].flags = 1;
		else if (buf[0] == '=')
			r->ent[i].flags = 4;
		else if (buf[0] == '+')
			r->ent[i].flags = 2;
		else if (buf[0] == 0)
			break;
		if (font_readgpat(f, r->ent + i, buf + 1))
			break;
	}
}
