#include <stdio.h>
#include <string.h>
#include "dev_font.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"

static void out_fontup(int f)
{
	void *font;
	char buf[64];

	if (o_m != p_m) {
		out("%d %d %d rgb\n", (o_m >> 16) & 0xff, (o_m >> 8) & 0xff, o_m & 0xff);
		p_m = o_m;
	}
	if (p_f == f && o_s == p_s)
		return;
	font = dev_font(f);
	out("%d /%s f\n", o_s, font_name(font));
	p_f = f;
	p_s = o_s;
	sprintf(buf, " %s ", font_name(font));
	if (!strstr(o_fonts, buf))
		sprintf(o_fonts + strlen(o_fonts), "%s ", font_name(font));
}
