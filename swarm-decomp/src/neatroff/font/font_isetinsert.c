static void font_isetinsert(struct font *f, struct iset *s, int i, struct fent *e)
{
	int *row;
	int *p;

	if (e->flags & 8) {
		row = iset_get(f->iac8, e->code);
		if (row == NULL)
			return;
		for (p = row; *p >= 0;)
			iset_put(s, *p++, i);
	} else if (e->code >= 0) {
		iset_put(s, e->code, i);
	}
}
