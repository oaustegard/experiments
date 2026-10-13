#include "ren_put.h"
#include "fn_charnext.h"
#include "cdef_map.h"
#include "fn_dev_font.h"
#include "fn_env_fmt.h"
#include "env_lc.h"
#include "env_tc.h"
#include "fn_env_wb.h"
#include "fn_eval.h"
#include "fn_font_swid.h"
#include "fmt_fillreq.h"
#include "fn_fmt_morelines.h"
#include "fn_fmt_wid.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_num.h"
#include "quotednext.h"
#include "unquotednext.h"
#include "ren_bcmd.h"
#include "ren_dcmd.h"
#include "ren_hlcmd.h"
#include "ren_ocmd.h"
#include "ren_vlcmd.h"
#include "ren_zcmd.h"
#include "wb_els.h"
#include "wb_etc.h"
#include "wb_flushdir.h"
#include "wb_hmov.h"
#include "wb_italiccorrection.h"
#include "wb_italiccorrectionleft.h"
#include "wb_put.h"
#include "wb_putexpand.h"
#include "wb_putraw.h"
#include "wb_setcost.h"
#include "wb_setpart.h"
#include "wb_vmov.h"
#include "fn_wb_wid.h"
#include "var_dev_res.h"
#include "c_ec.h"
#include "var_tr_c.h"
#include "ren_tab.h"
#include "ren_field.h"
#include "ren_ps.h"
#include "ren_cl.h"
#include "ren_ft.h"
#include "ren_passline.h"

static void ren_put(struct wb *w, char *s, int (*next)(void), void (*back)(int))
{
	char c;
	int d;
	char *p;
	char buf[0x80];
	int wid;
	void *f;

	c = *s;
	if (c == ' ' || c == '\n') {
		wb_put(w, s);
		return;
	}
	if ((c & ~8) == 1) {
		if (c == 9)
			ren_tab(w, env_tc(), next, back);
		else
			ren_tab(w, env_lc(), next, back);
		return;
	}
	if (c_fa[0] && strcmp(c_fa, s) == 0) {
		ren_field(w, next, back);
		return;
	}
	if (c == c_ec) {
		c = (unsigned char)s[1];
		if (c == 'z') {
			wid = wb_wid(w);
			if (charnext(buf, next, back) >= 0)
				ren_put(w, buf, next, back);
			wb_hmov(w, wid - wb_wid(w));
			return;
		}
		d = (char)c;
		if (!strchr(" bCcDdefHhjkLlmNoprSsuvXxZz0^|!{}&/,<>", d))
			goto plain;
		p = strchr("*fgkmns", d);
		if (p) {
			p = unquotednext(d, next, back);
			c = d = s[1];
		}
		if (strchr("bCDhHjlLNoRSvwxXZ?", d)) {
			p = quotednext(next, back);
			c = (unsigned char)s[1];
		}
		if (c == 'e') {
			snprintf(s, 0x20, "%c%c", c_ec, c_ec);
			free(p);
			goto plain;
		}
		if (c == 'N') {
			snprintf(s, 0x20, "GID=%s", p);
			free(p);
			goto plain;
		}
		switch (c) {
		case ' ':
			{
				int ss = *nreg(map(".ss"));
				int sz = *nreg('s');
				wb_hmov(w, font_swid(dev_font(*nreg('f')), sz, ss));
			}
			break;
		case '0':
			wb_hmov(w, zwid());
			break;
		case '/':
			wb_italiccorrection(w);
			break;
		case ',':
			wb_italiccorrectionleft(w);
			break;
		case '&':
			wb_hmov(w, 0);
			break;
		case '|':
			wb_hmov(w, *nreg('s') * dev_res / 864);
			break;
		case '^':
			wb_hmov(w, *nreg('s') * dev_res / 432);
			break;
		case 'Z':
			ren_zcmd(w, p);
			break;
		case 'b':
			ren_bcmd(w, p);
			break;
		case 'c':
			wb_setpart(w);
			break;
		case 'L':
			ren_vlcmd(w, p);
			break;
		case 'X':
			wb_etc(w, p);
			break;
		case 'v':
			wb_vmov(w, eval(p, 'v'));
			break;
		case 'x':
			wb_els(w, eval(p, 'v'));
			break;
		case 'h':
			wb_hmov(w, eval(p, 'm'));
			break;
		case 'd':
			wb_vmov(w, *nreg('s') * dev_res / 144);
			break;
		case 'f':
			ren_ft(p);
			break;
		case 'j':
			wb_setcost(w, eval(p, 0));
			break;
		case 'k':
			if (w == env_wb()) {
				wid = fmt_wid(env_fmt());
				wid += wb_wid(env_wb());
				wid -= *nreg(map(".b0"));
			} else {
				wid = wb_wid(w);
			}
			num_set(map(p), wid);
			break;
		case 'l':
			ren_hlcmd(w, p);
			break;
		case 'm':
			ren_cl(p);
			break;
		case 'o':
			ren_ocmd(w, p);
			break;
		case 'p':
			if (w == env_wb()) {
				for (;;) {
					if (!fmt_fillreq(env_fmt()))
						break;
					f = env_fmt();
					while (fmt_morelines(f))
						ren_passline(f);
				}
			}
			break;
		case 'r':
			wb_vmov(w, -(*nreg('s') * dev_res / 72));
			break;
		case 's':
			ren_ps(p);
			break;
		case 'u':
			wb_vmov(w, -(*nreg('s') * dev_res / 144));
			break;
		case 'D':
			ren_dcmd(w, p);
			break;
		case '<':
		case '>':
			*nreg(map(".cd")) = (c == '<');
			wb_flushdir(w);
			break;
		default:
			break;
		}
		free(p);
		return;
	}
plain:
	if (ren_div) {
		wb_putraw(w, s);
		return;
	}
	if (cdef_map(s, *nreg('f')) == 0)
		wb_put(w, s);
	else
		wb_putexpand(w, s);
}
