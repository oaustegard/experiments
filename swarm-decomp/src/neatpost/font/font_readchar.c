#include <stdio.h>
#include <string.h>
#include "font.h"
#include "fn_dict_put.h"
#include "fn_mextend.h"

static int font_readchar(struct font *f, FILE *fp, int *pn, int *pidx)
{
	int code;
	char a[32];
	char name[32];
	char b[128];
	struct fglyph *g;
	int n;
	int c;

	if (fscanf(fp, "%31s %127s", a, b) != 2)
		return 1;
	if (strcmp(a, "---") == 0)
		sprintf(a, "c%04d", *pn);
	if (strcmp(b, "\"") == 0) {
		dict_put(f->dict8a8, a, *pidx);
		return 0;
	}
	if (fscanf(fp, "%d %31s", &code, name) == 2) {
		c = code;
		if (f->nglyphs == f->cap) {
			f->cap = f->nglyphs + 0x400;
			f->glyphs = mextend(f->glyphs, f->nglyphs, f->cap, sizeof(struct fglyph));
		}
		g = f->glyphs + f->nglyphs;
		snprintf(g->name, 0x20, "%s", name);
		snprintf(g->key, 0x20, "%s", a);
		g->code0 = c;
		g->font = f;
		dict_put(f->dict898, g->name, f->nglyphs);
		n = f->nglyphs;
		f->nglyphs = n + 1;
		*pidx = n;
		g = f->glyphs + n;
		sscanf(b, "%d", &g->width);
		tilleol(fp, b);
		if (sscanf(b, "%d", &g->code) != 1)
			g->code = 0;
		dict_put(f->dict8a0, a, *pidx);
		(*pn)++;
		return 0;
	}
	return 1;
}
