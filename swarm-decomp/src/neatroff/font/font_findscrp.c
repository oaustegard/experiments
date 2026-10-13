static int font_findscrp(struct font *f, char *s)
{
	int i, r;

	i = 0;
	r = -1;
	while (f->scrpname[i][0] != 0) {
		if (strcmp(s, f->scrpname[i]) == 0) {
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
		snprintf(f->scrpname[i], 8, "%s", s);
	return i;
}
