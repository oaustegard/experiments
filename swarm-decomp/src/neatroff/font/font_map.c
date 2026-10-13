int font_map(void *f, char *s, char *t)
{
	struct font *F = f;
	int n = -1;

	if (t) {
		int i = dict_get(F->d78, t);

		n = -2;
		if (i >= 0 && F->glyph + i) {
			struct fglyph *p;

			i = dict_get(F->d78, t);
			p = i < 0 ? NULL : F->glyph + i;
			n = p - F->glyph;
		}
	}
	dict_put(F->d88, s, n);
	return 0;
}
