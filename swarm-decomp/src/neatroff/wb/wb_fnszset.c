#include "wb.h"
void wb_fnszset(struct wb *w, int a, int b, int c, int d)
{
	w->f20[0] = a;
	w->f20[1] = b;
	w->f20[2] = c;
	w->f20[3] = d;
}
