#include "array.h"
#include "sdef.h"
#include "ncc_data.h"
#include "type_totsz.h"

static int type_totsz(struct type *t)
{
	int n = 1;
	struct array *a;

	while (t->ptr == 0 && (t->flags & 1)) {
		a = (struct array *)arrays + t->arr;
		n *= a->size;
		t = &a->t;
	}
	if (t->flags & 2) {
		if (t->ptr)
			return n * 8;
		return n * ((struct sdef *)structs + t->arr)->pad[1];
	}
	if (t->ptr)
		return n * 8;
	return n * (t->kind & 0xf);
}
