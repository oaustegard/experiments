#include <stdlib.h>
#include "sbuf.h"

void sbuf_free(struct sbuf *sb)
{
	free(sb->s);
	free(sb);
}
