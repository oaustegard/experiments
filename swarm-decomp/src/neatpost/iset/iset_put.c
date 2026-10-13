#include <stdlib.h>
#include <string.h>
#include "iset.h"

void iset_put(struct iset *s, int i, int v)
{
	int *bp, *cp, k;

	if ((unsigned)i > 0xfffff)
		return;
	while (s->n <= i)
		iset_extend(s, (i + 0x400) & ~0x3ff);
	bp = s->b + i;
	cp = s->c + i;
	if (*cp + 1 >= *bp) {
		int nb = 2 * *bp + 8;
		int *p = malloc(nb * sizeof(int));

		if (s->a[i]) {
			memcpy(p, s->a[i], *bp * sizeof(int));
			free(s->a[i]);
		}
		*bp = nb;
		s->a[i] = p;
	}
	k = *cp;
	*cp = k + 1;
	s->a[i][k] = v;
	s->a[i][*cp] = -1;
}
