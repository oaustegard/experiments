#include <stdio.h>
#include <stdlib.h>

#include "c_ec.h"
#include "fn_nreg.h"
#include "fn_charnext.h"
#include "quotednext.h"
#include "utf8next.h"

int charnext(char *buf, int (*next)(void), void (*back)(void))
{
	int i, n;
	int ret;

	if (utf8next(buf, next) == 0)
		return -1;
	if (buf[0] == 4) {
		utf8next(buf + 1, next);
		return 4;
	}
	if (buf[0] != c_ec)
		return 0;
	utf8next(buf + 1, next);
	if (buf[1] == '(') {
		n = utf8next(buf, next);
		utf8next(buf + n, next);
		return '(';
	}
	n = *nreg('C');
	if (n == 0 && buf[1] == '[') {
		int c = next();
		if (c != '\n' && c != ']' && c >= 0) {
			i = 1;
			do {
				buf[i - 1] = c;
				n = i;
				c = next();
				if (c < 0 || c == '\n')
					break;
			} while (c != ']' && i++ <= 30);
		}
		buf[n] = 0;
		return '[';
	}
	ret = '\\';
	if (buf[1] == 'C') {
		char *s = quotednext(next, back);
		snprintf(buf, 32, "%s", s);
		free(s);
		ret = 'C';
	}
	return ret;
}
