#include "ren_put.h"
#include "fn_charnext.h"

static int ren_until(struct wb *w, int (*next)(void), void (*back)(int), char *s, char *t)
{
	char buf[0x80];
	int c;

	for (c = next(); c >= 0 && c != '\n'; c = next()) {
		back(c);
		if (charnext(buf, next, back) < 0)
			return -1;
		if (strcmp(s, buf) == 0)
			return 1;
		if (t && strcmp(t, buf) == 0)
			return 2;
		ren_put(w, buf, next, back);
	}
	if (c == '\n')
		back(c);
	return 0;
}
