struct fglyph *font_find(struct font *f, const char *name)
{
	struct fglyph *g = NULL;
	int i = dict_get(f->dict8a0, name);

	if (i < 0)
		i = dict_get(f->dict8a8, name);
	if (i >= 0)
		g = f->glyphs + i;
	return g;
}
