#include <stddef.h>
#include "font.h"
#include "fn_dict_get.h"

struct fglyph *font_glyph(struct font *f, const char *name)
{
	struct fglyph *g = NULL;
	int i = dict_get(f->dict898, name);
	if (i >= 0)
		g = f->glyphs + i;
	return g;
}
