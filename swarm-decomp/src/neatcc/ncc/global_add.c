static void global_add(struct decl *d)
{
	long n = globals_n;
	struct decl *tab = globals;
	struct decl *p, *np;

	if (n >= globals_sz) {
		int sz = globals_sz * 2;

		if (sz < 0x80)
			sz = 0x80;
		globals_sz = sz;
		np = malloc((long)sz * sizeof(struct decl));
		memcpy(np, tab, n * sizeof(struct decl));
		memset(np + n, 0, ((long)sz - n) * sizeof(struct decl));
		free(tab);
		globals = np;
		globals_n = n + 1;
		*(np + n) = *d;
		return;
	}
	globals_n = n + 1;
	*(tab + n) = *d;
}
