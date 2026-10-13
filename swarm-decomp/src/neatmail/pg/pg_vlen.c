#include <string.h>
#include "fn_hdrlen.h"

/* the length of a header value (at most 0x400 bytes), without trailing blanks */
static int pg_vlen(char *v)
{
	int vl = hdrlen(v, 0x400);
	while (vl > 0 && strchr(" \r\n", v[vl - 1]))
		vl--;
	return vl;
}
