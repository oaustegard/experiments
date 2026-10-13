#include "ext.h"

static void tr_tl(void)
{
	int c;

	for (;;) {
		c = cp_next();
		if (c < 0 || (c != ' ' && c != '\t'))
			break;
	}
	in_back(c);
	ren_tl(cp_next, in_back);
	for (;;) {
		c = cp_next();
		if (c < 0 || c == '\n')
			break;
	}
}
