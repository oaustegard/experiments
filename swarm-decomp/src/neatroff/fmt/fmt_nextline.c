#include "fmt.h"
#include "fn_sbuf_out.h"

char *fmt_nextline(struct fmt *f, int *p1, int *p2, int *p3, int *p4, int *p5, int *p6)
{
	struct fmtline *l;

	if (f->nlines == f->curline)
		return 0;
	l = &f->lines[f->curline++];
	*p2 = l->f14;
	*p3 = l->f1c;
	*p4 = l->f18;
	*p1 = l->f10;
	*p5 = l->f20;
	*p6 = l->f24;
	return sbuf_out(&l->buf);
}
