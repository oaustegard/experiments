#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_box_alloc.h"
#include "fn_box_over.h"
#include "fn_tok_poptext.h"
#include "fn_box_free.h"
#include "eqn_box.h"

static struct box *eqn_box(int sz, struct box *prev, int ps, char *text)
{
	unsigned int u = (unsigned short) sz;
	struct box *left, *right, *nb;
	int fl;

	left = eqn_left(sz, prev, ps, text);
	while (tok_jmp("over") == 0) {
		right = eqn_left(u | 1, NULL, ps, text);
		fl = 0;
		if (prev)
			fl = prev->flags;
		nb = box_alloc(ps, fl, u);
		printf(".ft %s\n", grfont);
		box_over(nb, left, right);
		box_free(left);
		left = nb;
		box_free(right);
	}
	return left;
}
