static struct decl *field_find(struct type *t, char *name)
{
	struct sdef *s;
	int i;

	s = (struct sdef *)structs + t->arr;
	for (i = 0; i < s->nfld; i++)
		if (strcmp(name, s->fld[i].name) == 0)
			return &s->fld[i];
	return NULL;
}
