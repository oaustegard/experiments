static void globalinit(struct decl *d, int local, struct type *t)
{
	char *name = d->pre[0] ? d->pre : d->name;
	char *s;
	char *b;
	int n;
	int m;

	if (t->flags & 1) {
		if (tok_grp() == '"') {
			struct array *a = (struct array *)arrays + t->arr;

			if (a->t.ptr == 0 && a->t.flags == 0 && (a->t.kind & 0xf) == 1) {
				s = tok_get();
				b = s + 1;
				n = tok_len();
				m = n - 2;
				b[m] = 0;
				o_dscpy(*(long *)d->loc + local, b, n - 1);
				return;
			}
		}
	}
	readexpr();
	ts_de(1);
	o_dsset(name, local, t->ptr ? 8 : t->kind);
	nts--;
}
