#include "iset.h"
#include "libc.h"
#include "fn_xmalloc.h"

void iset_put(void *set, int i, int v)
{
	struct iset *s = set;

	if ((unsigned)i > 0xfffff)
		return;
	if (s->n <= i)
		iset_extend(s, (i + 1024) & ~1023);
	if (s->n > i && s->rowlen[i] + 1 >= s->rowcap[i]) {
		int oldcap = s->rowcap[i];
		int cap = 2 * oldcap;
		int *p;

		if (!cap)
			cap = 8;
		p = xmalloc(cap * sizeof(int));
		if (s->tab[i]) {
			memcpy(p, s->tab[i], oldcap * sizeof(int));
			free(s->tab[i]);
		}
		s->rowcap[i] = cap;
		s->tab[i] = p;
	}
	s->tab[i][s->rowlen[i]++] = v;
	s->tab[i][s->rowlen[i]] = -1;
}
