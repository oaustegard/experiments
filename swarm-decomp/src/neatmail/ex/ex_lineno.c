#include <stdlib.h>
#include <ctype.h>
#include "fn_mbox_len.h"
#include "vars.h"
#include "ex_search.h"
#include "ex_lineno.h"

static int ex_lineno(char *s)
{
	int n;
	char c;

	n = pos;
	c = *s;
	if (isdigit(c)) {
		n = strtol(s, 0, 10);
		c = *s;
	}
	if (c == '$') {
		n = mbox_len(mbox) - 1;
		c = *s;
	}
	if (c == '-') {
		if (!s[1])
			return pos - 1;
		n = pos - ex_lineno(s + 1);
		c = *s;
	}
	if (c == '+') {
		if (!s[1])
			return pos + 1;
		n = ex_lineno(s + 1) + pos;
		c = *s;
	}
	if (c == '/') {
		if (s[1]) {
			n = ex_search(s);
			c = *s;
			if (c == '?' && s[1])
				return ex_search(s);
		}
		return n;
	}
	else if (c == '?' && s[1])
		return ex_search(s);
	return n;
}
