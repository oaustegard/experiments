#include "fn_map.h"
#include "fn_nreg.h"
#include "dir_state.h"
#include "dir_copy.h"

static void dir_flush(void)
{
	char *b = rbuf;

	if (*nreg(map(".td")) <= 0)
		b += rbuf_sz - rbuf_n;
	dir_copy(&dbuf, &dbuf_n, &dbuf_sz, b, rbuf_n, *nreg(map(".td")));
	rbuf_n = 0;
}
