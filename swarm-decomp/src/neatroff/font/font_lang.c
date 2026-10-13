void font_lang(void *vf, char *s)
{
	struct font *f = vf;
	int i = -1;

	if (s)
		i = font_findlang(f, s);
	f->lang = i;
}
