static int font_findfeat(struct font *f, char *s)
{
	int i, r;

	i = 0;
	r = -1;
	while (f->featname[i][0] != 0) {
		if (strcmp(s, f->featname[i]) == 0) {
			r = i;
			break;
		}
		i++;
		if (i == 128)
			break;
	}
	if (i == 128)
		return -1;
	if (r < 0)
		snprintf(f->featname[i], 8, "%s", s);
	return i;
}
