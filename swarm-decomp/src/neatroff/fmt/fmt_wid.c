#include "fmt.h"

static int fmt_wordgap(struct fmt *f);

int fmt_wid(void *p)
{
	struct fmt *f = p;
	int w = 0;
	struct fmtword *l;

	if (f->nwords) {
		l = &f->words[f->nwords - 1];
		w = l->pos - f->words[0].pos + f->words[0].wid + l->f18;
	}
	return fmt_wordgap(f) + w;
}
