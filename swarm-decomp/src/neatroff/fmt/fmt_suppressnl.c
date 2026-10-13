#include "fmt.h"

void fmt_suppressnl(struct fmt *f)
{
	if (f->nosuppress) {
		f->nosuppress--;
		f->f48 = 1;
	}
}
