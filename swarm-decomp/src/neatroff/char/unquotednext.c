#include <ctype.h>

#include "fn_nreg.h"
#include "fn_sbuf_add.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_out.h"
#include "sbuf.h"

char *unquotednext(int c, int (*next)(void), void (*back)(int))
{
	struct sbuf sb;
	int a, d;

	a = next();
	sbuf_init(&sb);
	if (c == 's' && (a == '+' || a == '-')) {
		sbuf_add(&sb, a);
		c = a;
		a = next();
	}
	if (a == '(') {
		sbuf_add(&sb, next());
		sbuf_add(&sb, next());
	} else if (*nreg('C') == 0 && a == '[') {
		d = next();
		while (d > 0 && d != '\n') {
			if (d == ']')
				break;
			sbuf_add(&sb, d);
			d = next();
		}
	} else {
		sbuf_add(&sb, a);
		if (*nreg('C') != 0) {
			a -= '1';
			if ((unsigned)a <= 2 && c == 's') {
				d = next();
				if (isdigit(d))
					sbuf_add(&sb, d);
				else
					back(d);
			}
		}
	}
	return sbuf_out(&sb);
}
