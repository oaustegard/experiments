void *font_find(void *f, char *s)
{
	struct font *F = f;
	int i = dict_get(F->d88, s);

	if (i == -1)
		i = dict_get(F->d80, s);
	return i < 0 ? NULL : F->glyph + i;
}
