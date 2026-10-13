#include <stdlib.h>
#include <string.h>

struct box *box_alloc(int type, int f24, int f28)
{
	struct box *b;

	b = malloc(sizeof *b);
	memset(b, 0, sizeof *b);
	sbuf_init(&b->buf);
	b->type = type;
	b->cnt = 0;
	b->f28 = f28;
	if (f24)
		b->f24 = f24;
	return b;
}
