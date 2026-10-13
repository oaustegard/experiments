#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_tok_get.h"
#include "fn_tok_poptext.h"
#include "fn_box_alloc.h"
#include "fn_box_matrix.h"
#include "fn_box_free.h"
#include "fn_errdie.h"
#include "eqn_boxuntil.h"

static void eqn_matrix(struct box *b, int ps, char *text)
{
	struct box *cells[32][32] = {{0}};
	int align[32];
	struct box **row;
	int vs, hs, ncol, c, n, i, j;

	vs = tok_jmp("{");
	if (vs != 0) {
		vs = strtol(tok_poptext(1), NULL, 10);
		tok_expect("{");
	}
	hs = 0;
	ncol = 0;
	for (;;) {
		if (tok_jmp("col") == 0)
			c = 'c';
		else if (tok_jmp("ccol") == 0)
			c = 'c';
		else {
			c = 'l';
			if (tok_jmp("lcol") != 0) {
				if (tok_jmp("rcol") != 0)
					break;
				c = 'r';
			}
		}
		align[ncol] = c;
		if (tok_jmp("{") != 0) {
			n = strtol(tok_poptext(1), NULL, 10);
			hs = hs >= n ? hs : n;
			tok_expect("{");
		}
		row = cells[ncol];
		for (;;) {
			if (tok_get() == NULL)
				errdie("neateqn: unterminated matrix col\n");
			*row++ = box_alloc(ps, 0, b->col);
			if (eqn_boxuntil(row[-1], ps, text, "above") != 0)
				break;
		}
		ncol++;
		tok_expect("}");
	}
	tok_expect("}");
	box_matrix(b, ncol, &cells[0][0], align, vs, hs);
	for (i = 0; i < ncol; i++)
		for (j = 0; j < 32; j++)
			if (cells[i][j] != NULL)
				box_free(cells[i][j]);
}
