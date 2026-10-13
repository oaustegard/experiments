static int array_add(struct type *t, int size)
{
	struct array *tab = arrays;
	struct array *p, *np;
	long n = arrays_n;

	if (n >= arrays_sz) {
		int sz = arrays_sz * 2;

		if (sz < 0x80)
			sz = 0x80;
		arrays_sz = sz;
		np = malloc((long)sz * sizeof(struct array));
		memcpy(np, tab, n * sizeof(struct array));
		memset(np + n, 0, (sz - n) * sizeof(struct array));
		free(tab);
		tab = np;
		arrays = tab;
	}
	arrays_n = n + 1;
	p = arrays + arrays_n - 1;
	p->t = *t;
	p->size = size;
	return p - tab;
}
