#include "ic_ent.h"
#include "ic_tab.h"

void o_back(long n)
{
	struct ic_ent *p;
	long end;

	if ((int)n < ic_n) {
		p = &ic[(int)n];
		end = (long)&ic[ic_n];
		do {
			if (p->op & 0x200)
				free(p->d);
			p++;
		} while ((long)p != end);
	}
	ic_n = n;
}
