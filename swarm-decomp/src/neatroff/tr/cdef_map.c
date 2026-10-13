#include "cdef.h"
#include "libc.h"

char *cdef_map(char *s, int f)
{
	int i, n = cdef_n;

	for (i = 0; i < n; i++) {
		if (cdef_fn[i] == f || cdef_fn[i] == 0) {
			if (strcmp(cdef_src[i], s) == 0) {
				if (cdef_expanding)
					break;
				return cdef_dst[i];
			}
		}
	}
	return 0;
}
