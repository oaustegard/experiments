#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_tok_poptext.h"
#include "fn_tok_get.h"
#include "fn_box_alloc.h"
#include "fn_box_pile.h"
#include "fn_box_free.h"
#include "fn_errdie.h"
#include "eqn_boxuntil.h"

static void eqn_pile(struct box *b, int ps, char *text, int align)
{
	struct box *items[32] = {0};
	struct box **p, **end;
	long n, next;
	int vs, r;

	vs = tok_jmp("{");
	if (vs != 0) {
		vs = strtol(tok_poptext(1), NULL, 10);
		tok_expect("{");
	}
	n = 0;
	for (;;) {
		if (tok_get() == NULL)
			errdie("neateqn: unterminated pile\n");
		items[n] = box_alloc(ps, 0, b->col);
		r = eqn_boxuntil(items[n], ps, text, "above");
		if (r != 0)
			break;
		n++;
	}
	tok_expect("}");
	box_pile(b, items, align, vs);
	end = &items[n + 1];
	p = items;
	do {
		box_free(*p++);
	} while (p != end);
}
