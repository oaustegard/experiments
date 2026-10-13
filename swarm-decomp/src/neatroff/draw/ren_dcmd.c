#include <ctype.h>
#include "wb.h"
#include "fn_eval_up.h"
#include "fn_wb_drawa.h"
#include "fn_wb_drawc.h"
#include "fn_wb_drawe.h"
#include "fn_wb_drawl.h"
#include "fn_wb_drawxbeg.h"
#include "fn_wb_drawxcmd.h"
#include "fn_wb_drawxdot.h"
#include "fn_wb_drawxend.h"

void ren_dcmd(struct wb *w, char *s)
{
	char buf[64];
	char *t, *u;
	int c, m, v, i, n;

	t = s + 1;
	c = *s;
	switch (tolower(c)) {
	case 'a':
		while (*t == ' ' || *t == '\t')
			t++;
		m = eval_up(&t, 'm');
		while (*t == ' ' || *t == '\t')
			t++;
		v = eval_up(&t, 'v');
		while (*t == ' ' || *t == '\t')
			t++;
		i = eval_up(&t, 'm');
		while (*t == ' ' || *t == '\t')
			t++;
		n = eval_up(&t, 'v');
		wb_drawa(w, c, m, v, i, n);
		return;
	case 'c':
		while (*t == ' ' || *t == '\t')
			t++;
		m = eval_up(&t, 'm');
		wb_drawc(w, c, m);
		return;
	case 'e':
		while (*t == ' ' || *t == '\t')
			t++;
		m = eval_up(&t, 'm');
		while (*t == ' ' || *t == '\t')
			t++;
		v = eval_up(&t, 'v');
		wb_drawe(w, c, m, v);
		return;
	case 'l':
		while (*t == ' ' || *t == '\t')
			t++;
		m = eval_up(&t, 'm');
		while (*t == ' ' || *t == '\t')
			t++;
		v = eval_up(&t, 'v');
		wb_drawl(w, c, m, v);
		return;
	case 'p':
	case '~':
		wb_drawxbeg(w, c);
		while (*t) {
			while (*t == ' ' || *t == '\t')
				t++;
			u = t;
			m = eval_up(&t, 'm');
			if (t != u) {
				while (*t == ' ' || *t == '\t')
					t++;
				u = t;
				v = eval_up(&t, 'v');
				if (t != u) {
					wb_drawxdot(w, m, v);
					continue;
				}
			}
			i = 0;
			while (*t & 0xdf) {
				buf[i++] = *t++;
				if (i == 63)
					break;
			}
			buf[i] = 0;
			wb_drawxcmd(w, buf);
		}
		wb_drawxend(w);
		return;
	default:
		return;
	}
}
