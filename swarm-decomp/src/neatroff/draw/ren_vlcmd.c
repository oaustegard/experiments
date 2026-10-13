#include "wb.h"
#include "fn_eval_up.h"
#include "fn_charread.h"
#include "fn_nreg.h"
#include "fn_wb_vmov.h"
#include "fn_wb_hmov.h"
#include "fn_wb_putexpand.h"
#include "var_dev_res.h"
#include "c_ec.h"

void ren_vlcmd(struct wb *w, char *s)
{
	char buf[32];
	char c;
	int n, len, wid, h, q, r, i;

	n = eval_up(&s, 'v');
	c = *s;
	if (c == c_ec && s[1] == '&') {
		s += 2;
		c = *s;
	}
	if (!c || charread(&s, buf) < 0) {
		strcpy(buf, "br");
	}
	if (n == 0)
		return;
	h = *nreg('s') * dev_res / 72;
	wid = cwid(buf);
	len = n;
	if (n < 0) {
		wb_vmov(w, n);
		len = -n;
	}
	if (h == 0) {
		q = 0;
		r = len;
		goto check;
	}
	if (h <= len) {
		goto divide;
	} else {
		wb_vmov(w, len / 2 - h);
		q = 1;
	}
loop:	for (i = 0; i < q; i++) {
		wb_vmov(w, h);
		wb_putexpand(w, buf);
		wb_hmov(w, -wid);
	}
	if (h > len)
		wb_vmov(w, len / 2);
	if (n < 0)
		wb_vmov(w, -len);
	wb_hmov(w, wid);
	return;
divide:
	q = len / h;
	r = len % h;
	if (r != 0)
		goto check;
	goto loop;
check:
	if (buf[1] == 0) {
		if (buf[0] == '|')
			goto special;
	} else if (buf[0] == 'b') {
		if (buf[1] == 'r' || buf[1] == 'v')
			goto special;
	}
	wb_vmov(w, r);
	goto loop;
special:
	wb_vmov(w, h);
	wb_putexpand(w, buf);
	wb_hmov(w, -wid);
	wb_vmov(w, r - h);
	goto loop;
}
