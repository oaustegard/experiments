void *font_glyph(void *f, char *s)
{
	struct font *F = f;
	int i = dict_get(F->d78, s);

	return i < 0 ? NULL : F->glyph + i;
}
