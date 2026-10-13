#include "sbuf.h"
#include "fn_sbuf_out.h"
#include "ext.h"
#include "read_name.h"

static char *read_name(int q)
{
	struct sbuf b;
	int c, n = 0, nq;

	c = cp_next();
	sbuf_init(&b);
	while (c >= 0 && c <= 0x20 && ((0x100000210ULL >> c) & 1))
		c = cp_next();
	if (c < 0)
		return sbuf_out(&b);
	if (c == 0 || c == '\n')
		goto back;
	nq = !q;
	for (;;) {
		if (c != 4) {
			n++;
			sbuf_add(&b, c);
		}
		c = cp_next();
		if (c <= 0 || c == ' ') {
			if (c < 0)
				return sbuf_out(&b);
			break;
		}
		if (c == 9 || c == '\n')
			break;
		if (n > 1 && !nq)
			break;
	}
back:
	in_back(c);
	return sbuf_out(&b);
}
