#include "fmt.h"

int fmt_morelines(struct fmt *f)
{
	return f->nlines != f->curline;
}
