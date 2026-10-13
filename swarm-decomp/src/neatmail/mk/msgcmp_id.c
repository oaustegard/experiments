#include <string.h>
#include "mkmsg.h"

static int msgcmp_id(const void *a, const void *b) {
	const struct mkmsg *x = *(struct mkmsg *const *)a;
	const struct mkmsg *y = *(struct mkmsg *const *)b;
	int yl = y->idlen;
	int xl = x->idlen;

	if (yl != xl)
		return yl - xl;
	return strncmp(x->id, y->id, yl);
}
