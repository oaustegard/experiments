#include "sbuf.h"

void
sbuf_init(struct sbuf *b)
{
	b->len = 0;
	/* volatile keeps gcc from fusing this with the len store into a movq */
	*(volatile int *)&b->size = 512;
	b->s = malloc(512);
}
