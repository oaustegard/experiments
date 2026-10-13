#include "tr_args.h"
#include "sbuf.h"
#include "fn_nreg.h"
#include "fn_sbuf.h"
#include "fn_cp_next.h"
#include "fn_cp_copymode.h"
#include "fn_in_back.h"
#include "read_name.h"
#include "mkargs_ds.h"

static void mkargs_ds(struct tr_args *a)
{
	struct sbuf b;
	char *s;
	int c, eol;

	s = read_name(*nreg('C'));
	sbuf_append((struct sbuf *)a, s);
	sbuf_add((struct sbuf *)a, 0);
	free(s);
	sbuf_init(&b);
	cp_copymode(1);
	while ((c = cp_next()) == ' ')
		;
	eol = c <= 0 || c == '\n';
	if (c == '"')
		c = cp_next();
	while (c > 0 && c != '\n') {
		if (c != 4)
			sbuf_add(&b, c);
		c = cp_next();
	}
	if (c >= 0)
		in_back(c);
	cp_copymode(0);
	if (!eol) {
		s = sbuf_out(&b);
		if (s) {
			sbuf_append((struct sbuf *)a, s);
			sbuf_add((struct sbuf *)a, 0);
			free(s);
		}
	} else {
		sbuf_done(&b);
	}
	for (;;) {
		c = cp_next();
		if (c < 0 || c == '\n')
			break;
	}
}
