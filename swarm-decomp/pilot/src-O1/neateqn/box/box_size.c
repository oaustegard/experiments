#include <stdio.h>

int box_size(struct box *b, char *s)
{
	int old = b->type;

	if (s == NULL || *s == '\0')
		return old;
	if (!b->nregd) {
		b->nregd = 1;
		b->type = nregmk();
	}
	if (*s == '-' || *s == '+')
		printf(".nr %s %s%s\n", nregname(b->type), nreg(old), s);
	else
		printf(".nr %s %s\n", nregname(b->type), s);
	return b->type;
}
