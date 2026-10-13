#include "tr_args.h"
#include "fn_dev.h"
#include "fn_font_feat.h"
static void tr_ff(struct tr_args *a)
{
	void *f = a->arg[0] ? dev_font(dev_pos(a->arg[0])) : a->arg[0];
	char **p;
	char *s;

	s = a->arg[1];
	p = &a->arg[2];
	if (!f) {
		if (!s)
			return;
		do
			p++;
		while (p[-1]);
		return;
	}
	while (s) {
		if (s[0] && s[1])
			font_feat(f, s + 1, s[0] == '+');
		s = *p++;
	}
}
