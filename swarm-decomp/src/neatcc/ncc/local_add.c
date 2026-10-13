static void local_add(struct decl *d)
{
	long n = locals_n;
	struct decl *tab = locals;
	struct decl *p, *np;

	if (n >= locals_sz) {
		int sz = locals_sz * 2;

		if (sz < 0x80)
			sz = 0x80;
		locals_sz = sz;
		np = malloc((long)sz * sizeof(struct decl));
		memcpy(np, tab, n * sizeof(struct decl));
		memset(np + n, 0, ((long)sz - n) * sizeof(struct decl));
		free(tab);
		locals = np;
		locals_n = n + 1;
		*(np + n) = *d;
		return;
	}
	locals_n = n + 1;
	*(tab + n) = *d;
}
