static void typedefdef(int local, struct decl *d, int storage)
{
	struct tdef *tab = typedefs;
	long n = typedefs_n;
	struct type *t = &d->t;
	struct tdef *p;

	if (n >= typedefs_sz) {
		int sz = typedefs_sz * 2;
		struct tdef *np;

		if (sz < 0x80)
			sz = 0x80;
		typedefs_sz = sz;
		np = malloc((long)sz * sizeof(struct tdef));
		memcpy(np, tab, n * sizeof(struct tdef));
		memset(np + n, 0, ((long)sz - n) * sizeof(struct tdef));
		free(tab);
		tab = np;
		typedefs = tab;
	}
	typedefs_n = n + 1;
	p = tab + typedefs_n - 1;
	strcpy(p->name, d->name);
	p->t = *t;
}
