#include "fn_sbuf_printf.h"
#include "pfont.h"
#include "sys_std.h"
static char *pdfcolor(int c)
{
	static char buf[64];
	int r = (c >> 16) & 255, g = (c >> 8) & 255, b = c & 255;
	int ri = r * 1000 / 255, gi = g * 1000 / 255, bi = b * 1000 / 255;
	sbuf_printf(pg, "%d.%03d %d.%03d %d.%03d", ri / 1000, ri % 1000, gi / 1000, gi % 1000, bi / 1000, bi % 1000);
	return buf;
}
