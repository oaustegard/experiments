#include "wb.h"
#include "fn_c_eos.h"

int wb_eos(struct wb *w)
{
	int i = w->f2068 - 1;

	while (i > 0 && c_eostran((char *)w + 0x68 + (long)i * 32))
		i--;
	if (i < 0)
		return 0;
	return c_eossent((char *)w + 0x68 + (long)i * 32) != 0;
}
