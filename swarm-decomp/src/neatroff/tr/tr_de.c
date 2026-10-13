#include <string.h>
#include "macrobody.h"
#include "sbuf.h"

static void tr_de(struct tr_args *a)
{
	struct sbuf b;
	int n;
	char *nm;

	if (a->arg[0]) {
		n = map(a->arg[0]);
		sbuf_init(&b);
		nm = *(char **)a;
		if (nm[1] == 'a' && nm[2] == 'm' && str_get(n))
			sbuf_append(&b, str_get(n));
		macrobody(&b, a->arg[1] ? a->arg[1] : ".");
		str_set(n, sbuf_buf(&b));
		sbuf_done(&b);
		if (!*nreg('C') && a->arg[2])
			str_dset(n, str_dget(map(a->arg[2])));
	}
}
