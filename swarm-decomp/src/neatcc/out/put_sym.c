#include <string.h>
#include "mextend.h"
#include "symstr_add.h"
#include "put_sym.h"
#include "symtab.h"

static struct sym *
put_sym(char *s)
{
	long i;
	int found = 0;
	struct sym *p;

	for (i = 0; i < syms_n; i++) {
		if (strcmp(s, symstr + syms[i].str) == 0) {
			found = 1;
			break;
		}
	}
	if (found)
		return &syms[i];
	if (syms_sz <= syms_n) {
		long sz = syms_sz * 2;
		if (sz < 0x80)
			sz = 0x80;
		syms_sz = sz;
		syms = mextend(syms, syms_n, sz, sizeof(struct sym));
	}
	p = &syms[syms_n++];
	p->str = symstr_add(s);
	p->type = 0x12;
	p->shndx = 0;
	return p;
}
