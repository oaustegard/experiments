#include "fn_charread.h"
#include "hcode_add.h"
#include "tr_args.h"

void tr_hcode(struct tr_args *a)
{
	char *s = a->arg[0];
	char from[32], to[32];

	while (s) {
		if (charread(&s, from) < 0)
			return;
		if (charread(&s, to) < 0)
			return;
		hcode_add(from, to);
	}
}
