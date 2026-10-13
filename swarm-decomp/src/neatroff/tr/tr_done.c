#include "cdef.h"
#include "cmap.h"
#include "fn_free.h"
#include "fn_dict_free.h"

/* release the cdef replacement strings and the request-name map at exit */
void tr_done()
{
	long n;
	char **p, **e;

	n = cdef_n;
	if (n > 0) {
		e = cdef_dst + n;
		for (p = cdef_dst; p != e; p++)
			free(*p);
	}
	dict_free(cmap);
}
