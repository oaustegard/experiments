#include <ctype.h>
#include "fn_escread.h"
#include "fn_hy.h"
#include "hw_add.h"
#include "hcode_mapchar.h"
#include "tr_args.h"

void tr_hw(struct tr_args *a)
{
	char buf[256], *tmp, *cur, *q, **p;

	p = a->arg;
	do {
		if (!*p)
			return;
		cur = *p;
		q = buf;
		for (;;) {
			if (escread(&cur, &tmp))
				break;
			if (*tmp == '-' && tmp[1])
				hcode_mapchar(tmp);
			else if (*tmp != '-')
				hcode_mapchar(tmp);
			q += hy_cput(q, tmp);
			if (q - buf > 0xdf)
				break;
		}
		hw_add(buf);
	} while (++p != a->arg + 31);
}
