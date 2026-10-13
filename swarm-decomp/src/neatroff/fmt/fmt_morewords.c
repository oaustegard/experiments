#include "fmt.h"

int fmt_morewords(struct fmt *f)
{
	if (f->nlines == f->curline)
		return f->nwords != 0;
	return 1;
}
