#include <string.h>
#include "fn_o_popnum.h"
#include "fn_o_popsym.h"
#include "fn_out_rel.h"
#include "fn_mem_cpy.h"
#include "fn_err.h"
void o_dsset(char *name, long off, long val)
{
	long i;
	long n = ds_n;
	long x;
	long b;
	long a;

	for (i = 0; i < n; i++)
		if (strcmp(name, ds_name + 128 * i) == 0) {
			off += (int)ds_off[i];
			break;
		}
	if (!o_popnum(&x)) {
		mem_cpy(&ds, off, (char *)&x, val & 0xf);
	} else if (!o_popsym(&a, &b)) {
		out_rel(a, 2, off);
		mem_cpy(&ds, off, (char *)&b, val & 0xf);
	} else {
		err("illegal assignment to static variables\n");
	}
}
