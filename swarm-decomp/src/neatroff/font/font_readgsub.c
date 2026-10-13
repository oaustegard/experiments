#include "font_gsub.h"
#include "font_readgpat.h"

static void font_readgsub(struct font *f, FILE *fp, int n, int a, int sel, int c, int d)
{
	struct frule *r;
	struct fent *e;
	char buf[128];
	int i;

	r = font_gsub(f, n, a, sel, c, d);
	for (i = 0; i < n; i++) {
		if (fscanf(fp, "%127s", buf) != 1)
			break;
		e = r->ent + i;
		if (buf[0] == '-')
			e->flags = 1;
		else if (buf[0] == '=')
			e->flags = 4;
		else if (buf[0] == '+')
			e->flags = 2;
		else if (buf[0] == 0)
			break;
		if (font_readgpat(f, e, buf + 1))
			break;
	}
}
