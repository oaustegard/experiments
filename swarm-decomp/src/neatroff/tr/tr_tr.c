#include "tr_args.h"
#include "fn_charread.h"
#include "fn_cmap_add.h"
#include "libc.h"

static void tr_tr(struct tr_args *a)
{
	char *s = a->arg[0];
	char from[32], to[32];

	while (s) {
		if (charread(&s, from) < 0)
			return;
		if (charread(&s, to) < 0)
			strcpy(to, " ");
		cmap_add(from, to);
	}
}
