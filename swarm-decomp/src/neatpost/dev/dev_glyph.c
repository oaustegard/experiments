#include <string.h>
#include "font.h"
#include "fglyph.h"
#include "fn_font_find.h"
#include "fn_font_glyph.h"

struct fglyph *dev_glyph(const char *s, int i)
{
	if (strncmp("GID=", s, 4) == 0)
		return font_glyph(fn_font[i], s + 4);
	return font_find(fn_font[i], s);
}
