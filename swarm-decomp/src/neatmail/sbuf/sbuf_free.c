#include <stdlib.h>
#include "sbuf.h"

void sbuf_free(struct sbuf *s)
{
	free(s->buf);
	free(s);
}
