#include "cdef.h"
#include "libc.h"

int cdef_expand(char *out, char *s, int f)
{
	int i, n = cdef_n;

	for (i = 0; i < n; i++) {
		if (cdef_fn[i] == f || cdef_fn[i] == 0) {
			if (strcmp(cdef_src[i], s) == 0) {
				int e = cdef_expanding;

				if (e)
					return 1;
				if (!cdef_dst[i])
					return 1;
				cdef_expanding = 1;
				ren_parse(out, cdef_dst[i]);
				cdef_expanding = 0;
				return e;
			}
		}
	}
	return 1;
}
