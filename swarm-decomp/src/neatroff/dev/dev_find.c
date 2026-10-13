#include "fn_font_find.h"
#include "fn_font_glyph.h"
#include "fn_font_special.h"
#include "dev_find.h"

static void *dev_find(char *s, int pos, int glyph)
{
	void *(*fn)(void *, char *);
	void *r;
	int i, k;

	fn = glyph ? font_glyph : font_find;
	r = fn(fn_font[pos], s);
	if (r)
		return r;
	for (i = 0; i < 4; i++) {
		k = fn_special[pos][i];
		if (k <= 0)
			break;
		r = fn((unsigned)k <= 31 ? fn_font[k] : NULL, s);
		if (r)
			return r;
	}
	for (i = 0; i < 32; i++) {
		if (fn_font[i] && font_special(fn_font[i])) {
			r = fn(fn_font[i], s);
			if (r)
				return r;
		}
	}
	return NULL;
}
