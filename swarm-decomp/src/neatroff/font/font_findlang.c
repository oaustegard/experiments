static int font_findlang(struct font *f, char *s)
{
	int i, r;

	i = 0;
	r = -1;
	while (f->langname[i][0] != 0) {
		if (strcmp(s, f->langname[i]) == 0) {
			r = i;
			break;
		}
		i++;
		if (i == 64)
			break;
	}
	if (i == 64)
		return -1;
	if (r < 0)
		snprintf(f->langname[i], 8, "%s", s);
	return i;
}
