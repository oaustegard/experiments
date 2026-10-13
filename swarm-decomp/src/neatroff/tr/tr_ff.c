#include "tr_args.h"
#include "fn_dev.h"
#include "fn_font_feat.h"
static void tr_ff(struct tr_args *a)
{
	void *f = 0;
	char **p;
	char *s;

	if (a->arg[0])
		f = dev_font(dev_pos(a->arg[0]));
	s = a->arg[1];
	if (!s)
		return;
	p = &a->arg[2];
	if (!f) {
		do
			p++;
		while (p[-1]);
	} else {
		do {
			if (!s[0] || !s[1])
				continue;
			font_feat(f, s + 1, s[0] == '+');
		} while ((s = *p++) != 0);
	}
}
