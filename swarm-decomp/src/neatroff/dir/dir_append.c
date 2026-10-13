#include <string.h>
#include "fn_map.h"
#include "fn_nreg.h"
#include "dir_state.h"
#include "dir_copy.h"
#include "dir_flush.h"

static void dir_append(char *s)
{
	int on = dir_cd > 0;

	if (*nreg(map(".td")) == on) {
		if (rbuf_n != 0)
			dir_flush();
	}
	if (*nreg(map(".td")) == on)
		dir_copy(&dbuf, &dbuf_n, &dbuf_sz, s, strlen(s), on);
	else
		dir_copy(&rbuf, &rbuf_n, &rbuf_sz, s, strlen(s), on);
}
