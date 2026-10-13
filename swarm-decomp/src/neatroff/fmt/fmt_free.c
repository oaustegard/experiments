#include "fmt.h"
#include "fn_free.h"

void fmt_free(void *p)
{
	struct fmt *f = p;
	int i;

	free(f->lines);
	for (i = 0; i < f->nwords; i++)
		free(f->words[i].s);
	free(f->words);
	free(f);
}
