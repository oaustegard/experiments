#include "wb.h"
#include "fn_wb_hmov.h"
#include "fn_wb_putexpand.h"
#include "c_ec.h"

void ren_hline(struct wb *w, int n, char *s)
{
	int wid, q, rem, i;
	char c;
	unsigned char d;

	wid = cwid(s);
	if (n < 0) {
		wb_hmov(w, n);
		n = -n;
	}
	if (wid == 0) {
		q = 0;
		rem = n;
	} else if (wid > n) {
		wb_hmov(w, (n - wid) / 2);
		wb_putexpand(w, s);
		q = 0;
		rem = 0;
	} else {
		q = n / wid;
		rem = n % wid;
	}
	if (rem != 0) {
		c = s[0];
		d = s[1];
		if (c == c_ec) {
			if (d == '_' || d == '-')
				goto special;
			goto normal;
		}
		if (d == 0) {
			if (c == '_')
				goto special;
			goto normal;
		}
		if (c == 'r') {
			if (d == 'n' || d == 'u')
				goto special;
			goto normal;
		}
		if (d == 'l' && c == 'u')
			goto special;
normal:
		wb_hmov(w, rem);
		goto loop;
special:
		rem -= wid;
		wb_putexpand(w, s);
		wb_hmov(w, rem);
	}
loop:
	for (i = 0; i < q; i++)
		wb_putexpand(w, s);
	if (wid > n)
		wb_hmov(w, (n - wid - 1) / 2);
}
