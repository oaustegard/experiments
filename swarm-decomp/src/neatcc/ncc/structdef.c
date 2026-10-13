#include "array.h"
#include "decl.h"
#include "sdef.h"
#include "ncc_data.h"
#include "type_totsz.h"

static void structdef(long si, struct decl *d)
{
	struct sdef *s = (struct sdef *)structs + si;
	struct type *t;
	int al, off, sz, pad1 = s->pad[1];	/* read once up front: gcc keeps it across the type_totsz calls */

	if (s->pad[0]) {
		*(long *)d->loc = 0;
		sz = type_totsz(&d->t);
		if (sz > pad1)
			s->pad[1] = sz;
	} else {
		t = &d->t;
		for (;;) {
			if (t->flags & 1) {
				if (t->ptr)
					break;
				t = &((struct array *)arrays + t->arr)->t;
				continue;
			}
			if (!(t->flags & 2) || t->ptr)
				break;
			t = &((struct sdef *)structs + t->arr)->fld[0].t;
		}
		al = type_totsz(t);
		if (al > 8)
			al = 8;
		if ((d->t.flags & 1) && d->t.ptr == 0) {
			al = type_totsz((struct type *)((struct array *)arrays + d->t.arr));
			if (al > 8)
				al = 8;
		}
		off = (pad1 + al - 1) & -al;
		*(long *)d->loc = off;
		s->pad[1] = off;
		sz = type_totsz(&d->t);
		s->pad[1] = off + sz;
	}
	s->fld[s->nfld++] = *d;
}
