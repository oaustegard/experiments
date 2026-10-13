#include <stdlib.h>
#include "sbuf.h"

void sbuf_free(struct sbuf *b)
{
	free(b->buf);
	free(b);
}
