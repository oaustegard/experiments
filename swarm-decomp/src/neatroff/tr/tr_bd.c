#include "tr_args.h"
#include "fn_dev.h"
#include "fn_eval.h"

static void tr_bd(struct tr_args *a)
{
	void *f;
	int n;

	if (!a->arg[0])
		return;
	f = dev_font(dev_pos(a->arg[0]));
	if (!a->arg[0])
		return;
	n = 'S' - (unsigned char)a->arg[0][0];
	if (n != 0)
		;
	else
		n = -(unsigned char)a->arg[0][1];
	if (!f || !n)
		return;
	font_setbd(f, a->arg[1] ? eval(a->arg[1], 'u') : 0);
}
