#include "cpp_protos.h"

static void buf_pop(void)
{
	int n = --bufs_n;
	if ((unsigned)bufs[n].kind <= 1)
		free(buf);
	if (n) {
		cur = bufs[n - 1].cur;
		len = bufs[n - 1].len;
		buf = bufs[n - 1].buf;
	}
}
