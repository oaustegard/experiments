#include "mkmsg.h"

static int msgcmp_date(const void *a, const void *b) {
	const struct mkmsg *x = *(struct mkmsg *const *)a;
	const struct mkmsg *y = *(struct mkmsg *const *)b;

	if (x->date == y->date)
		return x->idx - y->idx;
	return x->date - y->date;
}
