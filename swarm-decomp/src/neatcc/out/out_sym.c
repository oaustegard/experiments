#include "fn_out_sym.h"
#include "put_sym.h"
#include "symtab.h"

long
out_sym(char *s)
{
	return put_sym(s) - syms;
}
