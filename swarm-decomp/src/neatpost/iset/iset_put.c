#include <stdlib.h>
#include <string.h>
#include "iset.h"

void iset_put(struct iset *s, int i, int v)
{
	int *bp, *cp, *ap, k;

	if ((unsigned)i > 0xfffff)
		return;
	if (s->n <= i)
		goto extend;
main:
	cp = s->c + i;
	bp = s->b + i;
	k = *cp;
	if (k + 1 >= *bp)
		goto grow;
	ap = s->a[i];
store:
	*cp = k + 1;
	ap[k] = v;
	ap[*cp] = -1;
	return;
extend:
	iset_extend(s, (i + 0x400) & ~0x3ff);
	if (s->n > i)
		goto main;
	cp = s->c + i;
	ap = s->a[i];
	k = *cp;
	goto store;
grow: {
		int nb = 2 * *bp + 8;
		int *p = malloc(nb * sizeof(int));

		if (s->a[i]) {
			memcpy(p, s->a[i], *bp * sizeof(int));
			free(s->a[i]);
		}
		s->b[i] = nb;
		s->a[i] = p;
		ap = p;
		cp = s->c + i;
		k = *cp;
	}
	goto store;
}
