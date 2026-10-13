#include <string.h>
#include "mextend.h"
#include "symstr_add.h"
#include "put_sym.h"
#include "symtab.h"

static struct sym *
put_sym(char *s)
{
	long i;
	struct sym *p;

	for (i = 0; i < syms_n; i++) {
		if (strcmp(s, symstr + syms[i].str) == 0)
			break;
	}
	if (i < syms_n)
		return &syms[i];
	if (syms_n < syms_sz) {
	} else {
		long sz = syms_sz * 2;
		if (sz < 0x80)
			sz = 0x80;
		syms_sz = sz;
		syms = mextend(syms, syms_n, sz, sizeof(struct sym));
	}
	syms_n = syms_n + 1;
	p = &syms[syms_n - 1];
	p->str = symstr_add(s);
	p->type = 0x12;
	p->shndx = 0;
	return p;
}
