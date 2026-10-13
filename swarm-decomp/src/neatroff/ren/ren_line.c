#include "ren_line.h"
#include "ren_page.h"
#include "ren_pagelimit.h"
#include "ren_sp.h"

static int ren_line(char *p, int a, int b, int c, int d, int e, int f, int g, int h)
{
	struct sbuf s10, s20, s30, s40;
	char *b10, *b20, *b30;
	int w, cf, nd, t, v, d0;

	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	sbuf_init(&s10);
	sbuf_init(&s20);
	sbuf_init(&s30);
	sbuf_append(&s30, p);
	w = 1;
	if (*nreg('L') > 1)
		w = *nreg('L');
	w *= *nreg('v');
	d0 = *nreg('d');
	if (*nreg(map(".ns")) && !*p && !g && !h)
		goto done;
	if (g)
		ren_sp(-g, 1);
	ren_sp(0, 0);
	cf = c & 1;
	if (*p) {
		if (*nreg(map(".nm")) && cf)
			ren_lnum(&s10);
	}
	if (!ren_div) {
		if (dir_do) {
			sbuf_init(&s40);
			dir_fix(&s40, sbuf_buf(&s30));
			sbuf_done(&s30);
			sbuf_init(&s30);
			sbuf_append(&s30, sbuf_buf(&s40));
			sbuf_done(&s40);
		}
	}
	t = f - e - d;
	*nreg('n') = a;
	if ((b & 7) == 1) {
		if (a < t)
			d += (t - a) >> 1;
	} else {
		t = t - a + d;
		if ((b & 7) == 4)
			d = t;
	}
	if (d)
		sbuf_printf(&s10, "%ch'%du'", c_ec, d);
	if (cdiv && cdiv->f1c < a + d)
		cdiv->f1c = a + d;
	if (*p && cf) {
		if (*nreg(map(".mc")))
			ren_mc(&s20, a, d);
	}
	b20 = sbuf_buf(&s20);
	b30 = sbuf_buf(&s30);
	b10 = sbuf_buf(&s10);
	if (cdiv) {
		sbuf_append(&cdiv->buf, b10);
		sbuf_append(&cdiv->buf, b30);
		sbuf_append(&cdiv->buf, b20);
		sbuf_append(&cdiv->buf, "\n");
	} else {
		out("H%d\n", *nreg('o'));
		out("V%d\n", *nreg('d'));
		out_line(b10);
		out_line(b30);
		out_line(b20);
	}
	*nreg(map(".ns")) = 0;
	if (h)
		ren_sp(h, 1);
done:
	sbuf_done(&s10);
	sbuf_done(&s20);
	sbuf_done(&s30);
	*nreg('a') = h;
	nd = *nreg('d');
	t = trap_pos(d0);
	if (t >= 0 && (cdiv || t < *nreg('p')) && nd >= t)
		goto out_trap;
	v = *nreg('v');
	if (!cdiv) {
		t = w - v + *nreg(map("nl"));
		if (t >= *nreg('p'))
			goto out_trap;
	}
	if (!*p)
		return 0;
	if (*nreg('v') == w)
		return 0;
	return down(w - *nreg('v')) != 0;
out_trap:
	if (!ren_pagelimit(w - *nreg('v')))
		ren_traps(d0, *nreg('d'), 0);
	return 1;
}
