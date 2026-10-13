#include <stdlib.h>
#include "sbuf.h"

struct sbuf *sbuf_make(void)
{
	return calloc(16, 1);
}
