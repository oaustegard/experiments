#include <string.h>
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "fn_err.h"
#include "fn_o_popnum.h"
#include "readexpr.h"
#include "tok_grp.h"
#include "ts_de.h"
#include "type_totsz.h"
#include "sdef.h"
#include "array.h"
#include "decl.h"
#include "ncc_data.h"
#include "initexpr.h"

static void initexpr(struct type *t, int local, void *p, void (*fn)())
{
	struct sdef *s;
	struct decl *f, *cur, *end;
	struct type et, ct, *q;
	struct array *a;
	char *tk, *name;
	long i;
	int k, off, pp;

	if (t->ptr == 0 && (t->flags & 2)) {
		s = (struct sdef *)structs + t->arr;
		if (s->nfld > 0) {
			cur = s->fld;
			k = 0;
			do {
				tk = tok_see();
				if (*tk == '}' && !tk[1])
					break;
				tk = tok_see();
				if (*tk == '.' && !tk[1]) {
					tok_get();
					name = tok_get();
					s = (struct sdef *)structs + t->arr;
					end = s->fld + s->nfld;
					for (f = s->fld; f != end; f++)
						if (strcmp(name, f->name) == 0)
							break;
					if (f == end)
						err("unknown field <%s>\n", name);
					tk = tok_get();
					if (!(*tk == '=' && !tk[1]))
						err("syntax error (expected <%s> but got <%s>)\n", "=", tk);
				} else {
					f = cur;
				}
				off = f->loc[0] + local;
				tk = tok_see();
				if (*tk == '{' && !tk[1]) {
					tok_get();
					initexpr(&f->t, off, p, fn);
				} else {
					fn(p, off, &f->t);
				}
				tk = tok_see();
				if (!(*tk == ',' && !tk[1]))
					break;
				tok_get();
				cur++;
				k++;
			} while (k < ((struct sdef *)structs + t->arr)->nfld);
		}
		goto end;
	}
	if (!(t->flags & 1))
		goto end;
	a = (struct array *)arrays + t->arr;
	et = a->t;
	if (et.ptr == 0 && (et.kind & 0xf) == 1 && tok_grp() == '"') {
		fn(p, local, t);
		goto end;
	}
	i = 0;
	for (;;) {
		tk = tok_see();
		if (*tk == '}' && !tk[1])
			goto end;
		ct = et;
		tk = tok_see();
		if (*tk == '[' && !tk[1]) {
			tok_get();
			readexpr();
			ts_de(1);
			nts--;
			o_popnum(&i);
			tk = tok_get();
			if (!(*tk == ']' && !tk[1]))
				err("syntax error (expected <%s> but got <%s>)\n", "]", tk);
			tk = tok_get();
			if (!(*tk == '=' && !tk[1]))
				err("syntax error (expected <%s> but got <%s>)\n", "=", tk);
		}
		tk = tok_see();
		if (*tk == '{' && !tk[1]) {
			ct = et;
		} else if (tok_grp() == '"' && (ct.flags & 1)) {
			ct = et;
		} else {
			q = &et;
			pp = et.ptr;
			if (et.flags & 1) {
				while (pp == 0) {
					a = (struct array *)arrays + q->arr;
					q = &a->t;
					if (!(q->flags & 1))
						break;
					pp = q->ptr;
				}
			}
			ct = *q;
		}
		off = local + type_totsz(&ct) * (int)i;
		tk = tok_see();
		if (*tk == '{' && !tk[1]) {
			tok_get();
			initexpr(&ct, off, p, fn);
		} else {
			fn(p, off, &ct);
		}
		i++;
		tk = tok_see();
		if (!(*tk == ',' && !tk[1]))
			goto end;
		tok_get();
	}
end:
	tk = tok_get();
	if (!(*tk == '}' && !tk[1]))
		err("syntax error (expected <%s> but got <%s>)\n", "}", tk);
}
