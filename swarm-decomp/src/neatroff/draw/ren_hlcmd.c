#include "wb.h"
#include "fn_eval_up.h"
#include "c_ec.h"
#include "fn_charread.h"
#include "fn_ren_hline.h"

void ren_hlcmd(struct wb *w, char *s)
{
	char buf[32];
	char c;
	int n;

	n = eval_up(&s, 'm');
	c = *s;
	if (c == c_ec && s[1] == '&') {
		s += 2;
		c = *s;
	}
	if (!c || charread(&s, buf) < 0) {
		strcpy(buf, "ru");
	}
	if (n)
		ren_hline(w, n, buf);
}
