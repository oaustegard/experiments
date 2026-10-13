#include "cpp_macro_protos.h"

static long cpp_eval(void)
{
	char b[1024];
	long n;
	char *p;
	int limit;

	read_tilleol(b);
	buf_new(3, b, strlen(b));
	limit = bufs_limit;
	elen = 0;
	ecur = 0;
	bufs_limit = bufs_n;
	while (cpp_read(&p, &n) == 0) {
		memcpy(ebuf + elen, p, n);
		elen += n;
	}
	bufs_limit = limit;
	enext = -1;
	n = evalcexpr();
	buf_pop();
	return n;
}
