#include <string.h>
#include "array.h"
#include "ncc_data.h"
#include "type_totsz.h"
#include "type_szde.h"

static int type_szde(struct type *t)
{
	struct type x;
	struct array *a;

	x = *t;
	if (x.flags & 1) {
		if (x.ptr == 0) {
			a = (struct array *)arrays + x.arr;
			memcpy(&x, &a->t, sizeof(struct type));
			x.ptr++;
		}
	}
	x.ptr--;
	return type_totsz(&x);
}
