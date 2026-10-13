#include "fn_sbuf_printf.h"
#include "fn_dev_glyph.h"
#include "fn_dev_font.h"
#include "fn_font_path.h"
#include "fn_font_desc.h"
#include "fn_font_glnum.h"
#include "fn_font_wid.h"
#include "fn_font_swid.h"
#include "fn_mextend.h"
#include "font_name.h"
#include "fonttype.h"
#include "obj_map.h"
#include "writedesc.h"
#include "out_fontup.h"
#include "pdfpos00.h"

void outc(char *s)
{
	struct fglyph *g;
	struct font *f;
	struct pfont *e, *p;
	char *fname;
	int type, sub, n, i, j, v, id, glnum;
	static char buf[64];

	g = dev_glyph(s, o_f);
	if (g == 0) {
		void *df = dev_font(o_f);
		int w = 1;

		if (*s == ' ' && df != 0)
			w = font_swid(df, o_s);
		o_h += w;
		return;
	}
	f = g->font;
	fname = font_name(f);
	type = fonttype(font_path(f));
	sub = 0;
	if (type == '1')
		sub = font_glnum(f, g) / 256;
	p = pfonts;
	n = pfonts_n;
	for (i = 0; i < n; i++, p++)
		if (strcmp(fname, p->name) == 0 && p->f888 == sub)
			goto found;
	{
		if (pfonts_sz == n) {
			pfonts_sz = n + 16;
			pfonts = mextend(pfonts, n, pfonts_sz, sizeof(struct pfont));
		}
		e = pfonts + pfonts_n;
		snprintf(e->name, 0x80, "%s", fname);
		snprintf(e->path, 0x400, "%s", font_path(f));
		snprintf(e->desc, 0x400, "%s", font_desc(f));
		e->f894 = type == 't';
		id = obj_map();
		e->f888 = sub;
		e->objid = id;
		e->minc = 0x100000;
		n = pfonts_n;
		for (j = 0; j < n; j++)
			if (strcmp(pfonts[j].name, e->name) == 0)
				break;
		if (j < n)
			e->wid = pfonts[j].wid;
		else
			e->wid = writedesc(f);
		i = pfonts_n;
		pfonts_n = i + 1;
	}
found:
	o_i = i;
	o_iset[i] = 1;
	out_fontup();
	if (o_v != p_v) {
		tj_flush();
		sbuf_printf(pg, "1 0 0 1 %s Tm\n", pdfpos00(o_h * 100, o_v * 100));
		p_h = o_h;
		p_v = o_v;
	}
	if (o_queued == 0)
		sbuf_printf(pg, "[<");
	o_queued = 1;
	if (o_h != p_h) {
		v = (int)(((long)(p_h - o_h) * 72000 / o_s) / dev_res);
		sprintf(buf, "%s%d", v < 0 ? "-" : "", v < 0 ? -v : v);
		sbuf_printf(pg, "> %s <", buf);
	}
	glnum = font_glnum(g->font, g);
	if (pfonts[o_i].f894 != 0)
		sbuf_printf(pg, "%04x", glnum);
	else
		sbuf_printf(pg, "%02x", glnum % 256);
	if (glnum < pfonts[o_i].minc)
		pfonts[o_i].minc = glnum;
	if (glnum > pfonts[o_i].maxc)
		pfonts[o_i].maxc = glnum;
	p_h = o_h + font_wid(g->font, o_s, g->width);
}
