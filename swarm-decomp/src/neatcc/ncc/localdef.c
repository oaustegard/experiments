#include "fn_o_local.h"
#include "fn_o_num.h"
#include "fn_o_tmpdrop.h"
#include "fn_o_mklocal.h"
#include "fn_o_memset.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "type_totsz.h"
#include "initsize.h"
#include "initexpr.h"
#include "local_add.h"
#include "global_add.h"
#include "globaldef.h"
#include "readfunc.h"

static void localdef(int local, struct decl *d, int storage)
{
	struct array *ar;
	char *s;

	if (storage & 2) {
		global_add(d);
		return;
	}
	if ((d->t.flags & 4) && !d->t.ptr) {
		global_add(d);
		return;
	}
	if (storage & 1) {
		sprintf(d->pre, "__neatcc.%s.%s", func_name, d->name);
		globaldef(local, d, storage);
		return;
	}
	if ((d->t.flags & 1) && !d->t.ptr) {
		ar = (struct array *)arrays + d->t.arr;
		if (!ar->size)
			ar->size = initsize();
	}
	*(long *)d->loc = o_mklocal(type_totsz(&d->t));
	local_add(d);
	s = tok_see();
	if (s[0] != '=' || s[1])
		return;
	tok_get();
	if ((d->t.flags & 3) && !d->t.ptr) {
		o_local(*(long *)d->loc);
		o_num(0);
		o_num(type_totsz(&d->t));
		o_memset();
		o_tmpdrop(1);
	}
	s = tok_see();
	if (s[0] != '{' || s[1]) {
		localinit((long *)d->loc, 0, &d->t);
		return;
	}
	tok_get();
	initexpr(&d->t, 0, d->loc, localinit);
}
