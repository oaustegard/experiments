#include "cpp_protos.h"

static void buf_new(int kind, char *s, long n)
{
	int i = bufs_n;
	if (i) {
		struct buf *p = &bufs[i - 1];
		p->buf = buf;
		p->len = len;
		p->cur = cur;
		if (i > 31)
			die("nomem: NBUFS reached!\n");
	}
	buf = s;
	bufs_n = i + 1;
	cur = 0;
	len = n;
	bufs[i].kind = kind;
}
