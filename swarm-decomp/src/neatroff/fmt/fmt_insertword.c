#include <string.h>
#include "fmt.h"
#include "wb.h"
#include "env_hc.h"
#include "fn_wb_init.h"
#include "fn_wb_buf.h"
#include "fn_wb_cost.h"
#include "fn_wb_catstr.h"
#include "fn_wb_fnszget.h"
#include "fn_wb_fnszset.h"
#include "fn_wb_reset.h"
#include "fn_wb_swid.h"
#include "fn_wb_done.h"
#include "fn_escread.h"
#include "fn_c_hydash.h"

static void fmt_insertword(struct fmt *f, struct wb *w, int sp)
{
	char *s;
	int brk[32];
	int hyf[32] = {0};
	int spc[32] = {0};
	int n, i, r, bpl, last;
	int hy, a4, a5, a6;
	char *st, *en;
	int a, b, c, d;

	s = wb_buf(w);
	{
		char *p, *tok;

		tok = 0;
		p = s;
		do
			r = escread(&p, &tok);
		while (r > 0);
		if (r != 0)
			{ fmt_wb2word(f, w, 0, 1, sp, wb_cost(w)); return; };
		if (strcmp(env_hc(), tok) == 0)
			goto whole;;
		n = 0;
		last = 0;
	outer:
		bpl = n <= 31;
	inner:
		r = escread(&p, &tok);
		if (r < 0)
			goto d3;
		if (!bpl)
			goto d3;
		if (r > 0)
			goto inner;
		if (strcmp(env_hc(), tok) == 0) {
			hyf[n] = 1;
			brk[n] = p - s;
			n++;
		}
		if (c_hydash(tok)) {
			hyf[n] = 0;
			brk[n] = p - s;
			n++;
		}
		last = p - s;
		if (tok[0] == '\\' && tok[1] == '~' && tok[2] == 0) {
			spc[n] = 1;
			brk[n] = last;
			n++;
		}
		goto outer;
	d3:
		if (n <= 0)
			{ fmt_wb2word(f, w, 0, 1, sp, wb_cost(w)); return; };
		while (brk[n - 1] == last) {
			n--;
			if (n == 0)
				{ fmt_wb2word(f, w, 0, 1, sp, wb_cost(w)); return; };
		}
		if (f->fillreq == f->nwords + 1)
			f->fillreq += n;
	}
	{
		struct wb x;

		wb_init(&x);
		for (i = 0; i <= n;) {
			if (i < n)
				hy = hyf[i] != 0;
			else
				hy = 0;
			if (i == 0) {
				st = s;
				a4 = 1;
			} else {
				st = s + brk[i - 1];
				a4 = spc[i - 1] != 0;
			}
			if (i < n) {
				en = s + brk[i];
				if (spc[i])
					en -= 2;
			} else {
				en = s + strlen(s);
			}
			wb_catstr(&x, st, en);
			wb_fnszget(&x, &a, &b, &c, &d);
			if (i < n)
				a6 = spc[i] * 10000000;
			else
				a6 = wb_cost(w);
			if (i == 0)
				a5 = sp;
			else
				a5 = wb_swid(&x) * spc[i - 1];
			i++;
			fmt_wb2word(f, &x, hy, a4, a5, a6);
			wb_reset(&x);
			wb_fnszset(&x, a, b, c, d);
		}
		wb_done(&x);
		return;
	}
whole:
	fmt_wb2word(f, w, 0, 1, sp, wb_cost(w));
}
