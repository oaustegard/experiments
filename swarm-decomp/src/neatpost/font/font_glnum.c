#include "font.h"

int font_glnum(struct font *f, struct fglyph *g)
{
	return g - f->glyphs;
}
