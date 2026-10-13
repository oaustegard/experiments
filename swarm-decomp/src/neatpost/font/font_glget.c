#include <stddef.h>
#include "font.h"

struct fglyph *font_glget(struct font *f, int i)
{
	struct fglyph *g = NULL;
	if (i >= 0 && f->nglyphs > i)
		g = f->glyphs + i;
	return g;
}
