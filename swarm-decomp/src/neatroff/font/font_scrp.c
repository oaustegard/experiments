void font_scrp(void *vf, char *s)
{
	struct font *f = vf;
	int i = -1;

	if (s)
		i = font_findscrp(f, s);
	f->scrp = i;
}
