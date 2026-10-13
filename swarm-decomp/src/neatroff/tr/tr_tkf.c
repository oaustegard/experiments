#include "tr_args.h"
#include "fn_dev.h"
#include "fn_eval.h"
#include "fn_font_track.h"

static void tr_tkf(struct tr_args *a)
{
	void *f;
	int e4, e3, e2;

	if (!a->arg[0])
		return;
	f = dev_font(dev_pos(a->arg[0]));
	if (!f || !a->arg[4])
		return;
	e4 = eval(a->arg[4], 0);
	e3 = eval(a->arg[3], 0);
	e2 = eval(a->arg[2], 0);
	font_track(f, eval(a->arg[1], 0), e2, e3, e4);
}
