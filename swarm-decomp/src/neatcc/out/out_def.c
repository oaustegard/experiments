#include "fn_out_def.h"
#include "put_sym.h"

void
out_def(char *name, long flags, long a, long b)
{
	struct sym *p;
	long bind;
	long k = 1;

	p = put_sym(name);
	bind = (flags >> 4) & 1;
	if (flags & 1) {
		p->shndx = 1;
		k = 2;
	}
	if (flags & 2)
		p->shndx = 5;
	if (flags & 4)
		p->shndx = 7;
	p->type = (bind << 4) + k;
	p->val = a;
	p->size = b;
}
