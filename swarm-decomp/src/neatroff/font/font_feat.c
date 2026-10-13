int font_feat(void *vf, char *s, int plus)
{
	struct font *f = vf;
	int i, r = 0;

	i = font_findfeat(f, s);
	if (i >= 0) {
		r = f->feat[i];
		f->feat[i] = plus != 0;
	}
	return r;
}
