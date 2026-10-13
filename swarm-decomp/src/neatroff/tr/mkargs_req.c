#include "ext.h"

static void mkargs_req(struct sbuf *b)
{
	int c, n = 0;

	c = cp_next();
	do {
		while (c == '\t' || c == ' ')
			c = cp_next();
		if (c < 0 || c == '\n')
			break;
		while (c != ' ') {
			if (c != 4)
				sbuf_add(b, c);
			c = cp_next();
			if (c == 9 || c == 10 || c < 0)
				break;
		}
		n++;
		sbuf_add(b, 0);
		if (c == '\n')
			break;
		if (c < 0)
			break;
	} while (n != 32);
	if (c == '\n')
		in_back(c);
	c = cp_next();
	while (c >= 0 && c != '\n')
		c = cp_next();
}
