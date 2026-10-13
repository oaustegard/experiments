#include <string.h>
#include "mextend.h"
#include "symstr_add.h"
#include "strtab.h"
#include "symtab.h"

static long
symstr_add(char *s)
{
	long n;
	long len = strlen(s) + 1;
	long need;
	long sz;
	n = symstr_n;
	need = symstr_n + len;
	sz = symstr_sz;
	if (need < sz) {
	} else {
		sz = symstr_sz * 2;
		if (sz < 0x200)
			sz = 0x200;
		symstr_sz = sz;
		symstr = mextend(symstr, symstr_n, sz, 1);
		n = symstr_n;
		need = n + len;
	}
	n = symstr_n;
	strcpy(symstr + n, s);
	symstr_n = need;
	return n;
}
