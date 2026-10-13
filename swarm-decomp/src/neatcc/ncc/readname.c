#include "sys.h"
#include "func.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "fn_err.h"
#include "tok_grp.h"
#include "basetype.h"
#include "readarrays.h"
#include "readdefs.h"
#include "krdef.h"
#include "ncc_data.h"
#include "ncc_data_funcs.h"
#include "ncc_data_stmt.h"
#include "readname.h"

/* reads one declarator: out gets the declared type, name the identifier (if any).
   base, when given, is the type the declarator starts from; otherwise a base type is read.
   Returns nonzero when no base type is present. */
static int readname(struct type *out, char *name, struct type *base)
{
	struct type t;
	struct type saved;
	struct type ret;
	struct type ft = {0};
	struct type ps[32];
	char pn[32][0x80];
	int storage = 0;
	int np = 0;
	int variadic = 0;
	int isfunc = 0;
	char *s;

	if (name)
		name[0] = 0;
	if (base) {
		t = *base;
	} else if (basetype(&t, &storage)) {
		return 1;
	}
	for (;;) {
		s = tok_see();
		if (*s != '*' || s[1])
			break;
		tok_get();
		if (!t.kind)
			t.kind = 1;
		t.ptr++;
	}
	s = tok_see();
	if (s[0] == '(' && !s[1]) {
		saved = t;
		tok_get();
		for (;;) {
			s = tok_see();
			if (s[0] != '*' || s[1])
				break;
			tok_get();
			if (!t.kind)
				t.kind = 1;
			t.ptr++;
		}
		if (tok_grp() == 'a') {
			if (name)
				strcpy(name, tok_get());
		}
		readarrays(&t);
		s = tok_get();
		if (s[0] != ')' || s[1])
			err("syntax error (expected <%s> but got <%s>)\n", ")", s);
		s = tok_see();
		if (s[0] != '(' || s[1]) {
			if (readarrays(&saved)) {
				struct type *p = &t;
				int ptr = t.ptr;

				if (t.flags & 1) {
					for (;;) {
						ptr = p->ptr;
						if (ptr)
							break;
						p = &((struct array *)arrays)[p->arr].t;
						if (!(p->flags & 1)) {
							ptr = p->ptr;
							break;
						}
					}
				}
				*p = saved;
				p->ptr = ptr;
				p->tag = saved.tag;
			}
			goto done;
		}
		isfunc = 0;
	} else {
		if (tok_grp() == 'a') {
			if (name)
				strcpy(name, tok_get());
		}
		readarrays(&t);
		s = tok_see();
		if (s[0] != '(' || s[1])
			goto done;
		isfunc = 1;
	}
	s = tok_get();
	if (s[0] != '(' || s[1])
		err("syntax error (expected <%s> but got <%s>)\n", "(", s);
	for (;;) {
		s = tok_see();
		if (s[0] == ')' && !s[1])
			break;
		if (strcmp(tok_see(), "...") == 0) {
			tok_get();
			variadic = 1;
			break;
		}
		if (readname(&ps[np], pn[np], 0)) {
			ps[np].kind = 0x14;
			ps[np].flags = 0;
			ps[np].ptr = 0;
			ps[np].arr = 0;
			ps[np].tag = 0;
			strcpy(pn[np], tok_get());
		} else if ((ps[np].flags & 1) && !ps[np].ptr) {
			ps[np] = ((struct array *)arrays)[ps[np].arr].t;
			ps[np].ptr++;
		}
		np++;
		s = tok_see();
		if (s[0] != ',' || s[1])
			break;
		tok_get();
	}
	s = tok_get();
	if (s[0] != ')' || s[1])
		err("syntax error (expected <%s> but got <%s>)\n", ")", s);
	if (np == 1)
		np = (ps[0].kind | ps[0].ptr) != 0;
	ret = t;
	if (!isfunc)
		ret = saved;
	ft.kind = 8;
	ft.flags = 4;
	{
		struct func *f;
		int i;

		if (funcs_n >= funcs_sz) {
			int sz = funcs_sz * 2;

			if (sz < 0x80)
				sz = 0x80;
			funcs_sz = sz;
			void *nf = malloc((long)sz * sizeof(struct func));
			memcpy(nf, funcs, funcs_n * sizeof(struct func));
			memset((char *)nf + (long)funcs_n * sizeof(struct func), 0,
			       (sz - funcs_n) * sizeof(struct func));
			free(funcs);
			funcs = nf;
		}
		f = (struct func *)funcs + funcs_n;
		funcs_n++;
		f->param[32] = ret;
		for (i = 0; i < np; i++)
			f->param[i] = ps[i];
		f->nargs = np;
		f->flags = variadic;
		if (name)
			strcpy(f->name, name);
		else
			f->name[0] = 0;
		for (i = 0; i < np; i++)
			strcpy(f->arg[i], pn[i]);
		ft.arr = f - (struct func *)funcs;
	}
	if (isfunc) {
		t = ft;
	} else {
		struct type *p = &t;
		int ptr = t.ptr;

		if (t.flags & 1) {
			for (;;) {
				ptr = p->ptr;
				if (ptr)
					break;
				p = &((struct array *)arrays)[p->arr].t;
				if (!(p->flags & 1)) {
					ptr = p->ptr;
					break;
				}
			}
		}
		*p = ft;
		p->tag = ft.tag;
		p->ptr = ptr;
	}
	s = tok_see();
	if (s[0] == ';' && !s[1])
		goto done;
	for (;;) {
		s = tok_see();
		if (s[0] == '{' && !s[1])
			goto done;
		if (readdefs(krdef, t.arr))
			goto done;
		s = tok_get();
		if (s[0] == ';' && !s[1])
			continue;
		err("syntax error (expected <%s> but got <%s>)\n", ";", s);
		s = tok_see();
		if (s[0] == '{' && !s[1])
			goto done;
	}
done:
	*out = t;
	return 0;
}
