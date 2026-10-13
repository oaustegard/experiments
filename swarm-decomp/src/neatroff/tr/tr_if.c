#include "read_until.h"
static void tr_if(char **a)
{
	int c, neg, t, res;

	do
		c = cp_next();
	while (c == ' ' || c == '\t');
	if (c == '!') {
		c = cp_next();
		in_back(c);
		if (strchr("oetn", c) == 0) {
			if (c == ' ') {
				t = 1;
				goto end;
			}
			neg = 1;
			goto generic;
		}
		neg = 1;
	} else {
		neg = 0;
		in_back(c);
	}
	if (strchr("oetn", c) == 0)
		goto generic;
	c = cp_next();
	if (c == 'o') {
		t = neg != *nreg(map("%")) % 2;
		goto end;
	}
	if (c == 't') {
		t = neg ^ 1;
		goto end;
	}
	if (c == 'e') {
		t = neg ^ (~*nreg(map("%")) & 1);
		goto end;
	}
	t = neg;
	goto end;
generic:
	if (isdigit(c) || strchr("-+*/%<=>&:.|()", c)) {
		struct sbuf b;

		sbuf_init(&b);
		read_until(&b, 0, cp_next, in_back);
		res = eval(sbuf_buf(&b), 0) > 0;
		sbuf_done(&b);
	} else {
		char buf[32];
		struct sbuf s1, s2;

		charnext(buf, cp_next, in_back);
		sbuf_init(&s1);
		sbuf_init(&s2);
		read_until(&s1, buf, cp_next, in_back);
		read_until(&s2, buf, cp_next, in_back);
		cp_reqbeg();
		res = strcmp(sbuf_buf(&s1), sbuf_buf(&s2)) == 0;
		sbuf_done(&s1);
		sbuf_done(&s2);
	}
	t = neg ^ res;
end:
	if (a[0][1] == 'i' && a[0][2] == 'e' && ie_depth <= 127) {
		ie_cond[ie_depth] = t;
		ie_depth++;
	}
	cp_blk(!t);
}
