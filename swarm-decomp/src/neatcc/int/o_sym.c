#include "ic_put.h"
#include "fn_out_sym.h"

/* append a symbol entry for s (op 0x400400) and push its index on iv */
void o_sym(char *s)
{
	ic_put(0x400400, out_sym(s), 0, 0);
}
