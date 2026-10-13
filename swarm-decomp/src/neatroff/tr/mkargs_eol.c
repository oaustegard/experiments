#include "ext.h"

static void mkargs_eol(struct sbuf *b)
{
	int c;

	cp_copymode(1);
	while ((c = cp_next()) == ' ')
		;
	while (c >= 0 && c != '\n') {
		if (c != 4)
			sbuf_add(b, c);
		c = cp_next();
	}
	cp_copymode(0);
}
