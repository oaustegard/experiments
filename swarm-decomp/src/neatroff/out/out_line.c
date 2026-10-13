#include <ctype.h>
#include "libc.h"
#include "c_ec.h"
#include "fn_out.h"
#include "fn_escread.h"
#include "fn_hy.h"
#include "fn_cmap_map.h"
#include "fn_dev_glyph.h"
#include "fn_dev_font.h"
#include "fn_dev_pos.h"
#include "fn_dev_fontpos.h"
#include "fn_font_gwid.h"
#include "fn_font_wid.h"
#include "fn_font_mapped.h"
#include "fn_font_getcs.h"
#include "fn_font_getbd.h"
#include "fn_font_zoom.h"
#include "fn_eval.h"
#include "fn_eval_up.h"
#include "fn_nreg.h"
#include "fn_clr_get.h"
#include "fn_clr_str.h"

int font_wid(void *, int, int);

#define GFONT(g) (*(void **)((char *)(g) + 0x40))
#define GSIZE(g) (*(short *)((char *)(g) + 0x48))
#define GNAME(g) ((char *)(g) + 0x20)

void out_line(char *s) {
	char *p, *q, *b, *m, *e;
	void *g, *f;
	long c;
	unsigned char k;
	int r, n, x, y, w, zoom, pos, r13, r14;

	for (;;) {
		r = escread(&s, &p);
		if (r < 0)
			return;
		if (r != 0) {
			switch (r) {
			case 'D':
				q = p + 1;
				c = *p;
				out("D%c", c);
				switch (tolower(c)) {
				case 'a':
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'm'));
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'v'));
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'm'));
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'v'));
					outnn("\n");
					break;
				case 'c':
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'm'));
					outnn("\n");
					break;
				case 'e':
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'm'));
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'v'));
					outnn("\n");
					break;
				case 'l':
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'm'));
					while (*q == ' ' || *q == '\t') q++;
					outnn(" %d", eval_up(&q, 'v'));
					outnn(" .");
					outnn("\n");
					break;
				case 'p':
				case '~':
					b = q;
					while (*b) {
						while (*q == ' ' || *q == '\t') q++;
						b = q;
						x = eval_up(&q, 'm');
						e = q;
						if (e == b)
							goto nopar;
						while (*q == ' ' || *q == '\t') q++;
						e = q;
						y = eval_up(&q, 'v');
						if (q == e)
							goto nopar;
						outnn(" %d", x);
						outnn(" %d", y);
						b = q;
						continue;
					nopar:
						outnn(" ");
						while ((*q & 0xdf) != 0) {
							c = *q;
							q++;
							outnn("%c", c);
						}
						b = q;
					}
					outnn("\n");
					break;
				default:
					outnn("\n");
					break;
				}
				break;
			case 'X':
				out("x X %s\n", p);
				break;
			case 'f':
				n = dev_pos(p);
				if (n >= 0 && n != o_f) {
					o_f = n;
					out("f%d\n", n);
				}
				break;
			case 'h':
				outnn("h%d", eval(p, 'm'));
				break;
			case 'm':
				if (*nreg('C') == 0) {
					n = clr_get(p);
					if (n >= 0 && n != o_m) {
						o_m = n;
						out("m%s\n", clr_str(n));
					}
				}
				break;
			case 's':
				n = eval(p, 0);
				if (n != o_s) {
					o_s = n;
					out("s%d\n", n);
				}
				break;
			case 'v':
				outnn("v%d", eval(p, 'v'));
				break;
			}
			continue;
		}
		k = (unsigned char)*p;
		if (k == 4 || (k == '\\' && p[1] == '\\')) {
			p[0] = p[1];
			p[1] = 0;
			k = (unsigned char)*p;
		}
		if ((k & ~8) == 1)
			continue;
		if (c_hymark(p))
			continue;
		m = cmap_map(p);
		g = dev_glyph(m, o_f);
		f = dev_font(o_f);
		if (!g)
			continue;
		r14 = GSIZE(g);
		r13 = o_s;
		r13 = font_gwid(GFONT(g), dev_font(o_f), r13, r14);
		r14 = font_wid(GFONT(g), o_s, GSIZE(g));
		if (font_mapped(GFONT(g), m))
			m = GNAME(g);
		if (font_getcs(f))
			outnn("h%d", (r13 - r14) / 2);
		zoom = font_zoom(GFONT(g), o_s);
		pos = dev_fontpos(GFONT(g));
		outg(m, pos, zoom);
		if (font_getbd(f)) {
			outnn("h%d", font_getbd(f) - 1);
			zoom = font_zoom(GFONT(g), o_s);
			pos = dev_fontpos(GFONT(g));
			outg(m, pos, zoom);
			outnn("h%d", 1 - font_getbd(f));
		}
		if (font_getcs(f))
			outnn("h%d", (r14 - r13) / 2);
		outnn("h%d", r13);
	}
}
