static int font_readgpat(struct font *f, struct fent *e, unsigned char *s)
{
	struct fglyph *g;
	int i;

	if (isdigit(*s)) {
		e->code = strtol(s, NULL, 10);
	} else if (*s == '@') {
		e->code = strtol(s + 1, NULL, 10);
		if (iset_len(f->iac8, e->code) == 1)
			e->code = *iset_get(f->iac8, e->code);
		else
			e->flags |= 8;
	} else {
		i = dict_get(f->d78, s);
		if (i < 0)
			goto none;
		g = f->glyph + i;
		if (g == NULL)
			goto none;
		e->code = g - f->glyph;
	}
	return e->code < 0 || f->nglyph <= e->code;
none:
	e->code = -1;
	return 1;
}
