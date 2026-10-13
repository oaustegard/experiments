#include "wb.h"
#include "wb_flushsub.h"

void wb_wconf(struct wb *w, int *a, int *b, int *c, int *d, int *e, int *f, int *g)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	*a = w->a48[0];
	*b = -w->a48[2];
	*c = -w->a48[1];
	*d = w->a54[0] >= 0x20000000 ? 0 : w->a54[0];
	*e = w->a54[1] >= 0x20000000 ? 0 : -w->a54[1];
	*f = w->a54[2] <= (int)0xe0000000 ? 0 : w->a54[2];
	*g = w->a54[3] > (int)0xe0000000 ? -w->a54[3] : 0;
}
