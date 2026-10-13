#include "wb.h"
#include "fn_c_eos.h"

int wb_eos(struct wb *w)
{
	int i = w->f2068 - 1;
	char *p;

	if (i > 0) {
		p = (char *)w + 0x68 + (long)i * 32;
		while (c_eostran(p)) {
			p -= 32;
			i--;
			if (i <= 0)
				break;
		}
	}
	if (i < 0)
		return 0;
	return c_eossent((char *)w + 0x68 + (long)i * 32) != 0;
}
