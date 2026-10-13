static void font_readfeat(struct font *f, char *s, int *feat, int *scrp, int *lang)
{
	char *p, *q, *r;

	*scrp = -1;
	*lang = -1;
	p = strchr(s, ':');
	if (p) {
		*p++ = 0;
		q = p;
		r = strchr(q, ':');
		if (r) {
			*r = 0;
			*scrp = font_findscrp(f, q);
			*lang = font_findlang(f, r + 1);
		} else {
			*scrp = font_findscrp(f, q);
		}
	}
	*feat = font_findfeat(f, s);
}
