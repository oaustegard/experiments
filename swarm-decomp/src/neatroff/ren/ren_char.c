#include "ren_put.h"
#include "fn_charnext.h"

int ren_char(struct wb *w, int (*next)(void), void (*back)(int))
{
	char buf[0x80];

	if (charnext(buf, next, back) < 0)
		return -1;
	ren_put(w, buf, next, back);
	return 0;
}
