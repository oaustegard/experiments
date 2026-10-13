#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_tok_get.h"
#include "fn_box_merge.h"
#include "fn_box_free.h"
#include "eqn_box.h"
#include "eqn_boxuntil.h"

static int eqn_boxuntil(struct box *b, int ps, char *text, char *until)
{
	struct box *x = NULL;
	char *t;

	for (;;) {
		if (tok_get() == NULL)
			return 0;
		if (tok_jmp(until) == 0)
			return 0;
		t = tok_get();
		if (strcmp("}", t) == 0)
			return 1;
		x = eqn_box(b->col, x ? b : NULL, ps, text);
		box_merge(b, x, 0);
		box_free(x);
	}
}
