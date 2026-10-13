#include <string.h>
#include "outf.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"
#include "dev_font.h"

void out_fontup(int f);

void outc(const char *s)
{
	struct fglyph *g;
	void *f;
	const char *str;
	int type, c, n, v;

	g = dev_glyph(s, o_f);
	if (!g) {
		f = dev_font(o_f);
		n = 1;
		if (*s == ' ' && f)
			n = font_swid(f, o_s);
		o_h += n;
		return;
	}
	out_fontup(dev_fontid(g->font));
	if (g->code <= 0)
		type = 2;
	else
		type = 2 - (o_gname == 0);
	if (type != o_qtype)
		goto O;
	if (o_qend != o_h)
		goto O;
	if (o_qv != o_v)
		goto O;
	goto E;
O:
	o_flush();
	o_qh = o_h;
	str = "(";
	if (type != 1) {
		type = 2;
		str = "[";
	}
	o_qv = o_v;
	o_qtype = type;
	outf(str);
E:
	if (o_qtype != 1)
		goto P;
	c = g->code;
	if ((unsigned)(c - 32) > 94)
		outf("\\%d%d%d", (c >> 6) & 7, (c >> 3) & 7, c & 7);
	else
		outf("%s%c", strchr("()\\", c) ? "\\" : "", c);
	goto W;
P:
	outf("/%s", g->name);
W:
	o_qend = o_h + font_wid(g->font, o_s, g->width);
}
