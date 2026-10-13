#include <string.h>
#include "fn_charnext.h"
#include "fn_charnext_str.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_out.h"

char *quotednext(int (*next)(void), void (*back)(void)) {
	struct sbuf sb;
	char a[32], b[32], t[32];
	int c;
	charnext(a, next, back);
	sbuf_init(&sb);
	c = charnext(b, next, back);
	while (strcmp(b, a) != 0) {
		if (c < 0)
			return sbuf_out(&sb);
		charnext_str(t, b);
		sbuf_append(&sb, t);
		c = charnext(b, next, back);
	}
	return sbuf_out(&sb);
}
